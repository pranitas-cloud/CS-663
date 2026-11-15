"""Tests for the preprocessing pipelines using a tiny synthetic MVTec-style tree.

Run from the repo root:  pytest tests/test_preprocessing.py -q
The fake data mimics the real audit findings: square images of different sizes,
RGB and single-channel ('L') categories, and {0, 255} masks.
"""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.pipelines import (
    AutoencoderPreprocessor,
    PatchCorePreprocessor,
    PCAPreprocessor,
    build_dataset,
    build_preprocessor,
    stack_images,
)
from src.data.preprocessing import (
    extract_patches,
    fold_patches,
    image_to_tensor,
    mask_to_tensor,
    patch_grid,
    resize_image,
)


def _save(path, arr, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)


def _make_category(root, name, size, mode, n_train=10):
    rng = np.random.default_rng(0)
    base = root / name
    shape = (size, size, 3) if mode == "RGB" else (size, size)
    for i in range(n_train):
        _save(base / "train" / "good" / f"{i:03d}.png",
              rng.integers(0, 255, shape, dtype=np.uint8), mode)
    for i in range(2):
        _save(base / "test" / "good" / f"{i:03d}.png",
              rng.integers(0, 255, shape, dtype=np.uint8), mode)
    for i in range(2):
        _save(base / "test" / "crack" / f"{i:03d}.png",
              rng.integers(0, 255, shape, dtype=np.uint8), mode)
        mask = np.zeros((size, size), np.uint8)
        mask[size // 4: size // 2, size // 4: size // 2] = 255
        _save(base / "ground_truth" / "crack" / f"{i:03d}_mask.png", mask, "L")


@pytest.fixture()
def cfg(tmp_path):
    _make_category(tmp_path, "fake_rgb", 700, "RGB")
    _make_category(tmp_path, "fake_gray", 1024, "L")
    return {
        "seed": 42,
        "data": {"root": str(tmp_path), "category": "fake_rgb", "val_fraction": 0.1,
                 "image_size": 256, "mask_resize": "nearest"},
        "pca": {"image_size": 256, "grayscale": True, "patch_size": 32, "stride": 16},
        "autoencoder": {"image_size": 256, "grayscale": False},
        "patchcore": {"image_size": 256},
    }


@pytest.mark.parametrize("category", ["fake_rgb", "fake_gray"])
def test_output_shapes_and_ranges(cfg, category):
    shapes = {"pca": 1, "autoencoder": 3, "patchcore": 3}
    for method, channels in shapes.items():
        ds = build_dataset(cfg, method, "test", category)
        item = ds[0]
        assert item["image"].shape == (channels, 256, 256)
        assert item["mask"].shape == (1, 256, 256)
        assert item["image"].dtype == torch.float32
    pca_img = build_dataset(cfg, "pca", "train", category)[0]["image"]
    ae_img = build_dataset(cfg, "autoencoder", "train", category)[0]["image"]
    assert 0.0 <= pca_img.min() and pca_img.max() <= 1.0
    assert 0.0 <= ae_img.min() and ae_img.max() <= 1.0


def test_gray_source_replicated_for_rgb_methods(cfg):
    ae = build_dataset(cfg, "autoencoder", "train", "fake_gray")[0]["image"]
    assert torch.allclose(ae[0], ae[1]) and torch.allclose(ae[1], ae[2])


def test_gray_source_unchanged_for_pca():
    arr = np.random.default_rng(1).integers(0, 255, (256, 256), dtype=np.uint8)
    t = image_to_tensor(arr, grayscale=True)
    assert torch.allclose(t[0], torch.from_numpy(arr).float() / 255.0)


def test_patchcore_uses_imagenet_stats(cfg):
    img = build_dataset(cfg, "patchcore", "train", "fake_rgb")[0]["image"]
    # uniform noise in [0,1] standardised with ImageNet stats -> roughly (-2.1, 2.7)
    assert img.min() < -1.5 and img.max() > 1.5


def test_masks_are_binary_and_normals_are_empty(cfg):
    ds = build_dataset(cfg, "pca", "test", "fake_rgb")
    for i in range(len(ds)):
        item = ds[i]
        assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}
        if item["label"] == 0:
            assert item["mask"].sum() == 0
        else:
            assert item["mask"].sum() > 0
    assert tuple(ds[0]["orig_size"].tolist()) == (700, 700)


def test_mask_max_mode_keeps_tiny_defect():
    mask = np.zeros((1024, 1024), np.uint8)
    mask[501, 501] = 255  # one source pixel
    kept = mask_to_tensor(mask, 256, mode="max").sum()
    assert kept >= 1


def test_loader_collates_mixed_batch(cfg):
    ds = build_dataset(cfg, "autoencoder", "test", "fake_rgb")
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch["image"].shape == (4, 3, 256, 256)
    assert batch["mask"].shape == (4, 1, 256, 256)
    assert batch["label"].shape == (4,)


def test_split_is_normal_only_and_disjoint(cfg):
    tr = build_dataset(cfg, "pca", "train", "fake_rgb")
    va = build_dataset(cfg, "pca", "val", "fake_rgb")
    assert not ({s[0] for s in tr.samples} & {s[0] for s in va.samples})
    assert all(s[2] == "good" for s in tr.samples + va.samples)
    assert stack_images(tr).shape == (len(tr), 1, 256, 256)


def test_antialiasing_suppresses_aliasing():
    stripes = torch.zeros(1, 1024, 1024)
    stripes[:, :, ::2] = 1.0  # finest possible pattern, mean 0.5
    out = resize_image(stripes, 256)
    assert abs(out.mean().item() - 0.5) < 0.02
    assert out.std().item() < 0.05  # naive subsampling would give std 0.5


def test_patch_roundtrip_is_identity():
    x = torch.rand(2, 1, 256, 256)
    p = extract_patches(x, 32, 16)
    assert p.shape == (2, 15 * 15, 32 * 32)
    assert patch_grid(256, 32, 16) == (15, 15)
    back = fold_patches(p, 1, 256, 32, 16)
    assert torch.allclose(back, x, atol=1e-6)


def test_patch_grid_rejects_non_tiling_stride():
    with pytest.raises(ValueError):
        patch_grid(256, 32, 20)


def test_non_square_and_bad_dtype_rejected():
    pre = PatchCorePreprocessor(256)
    with pytest.raises(ValueError):
        pre(np.zeros((300, 200, 3), np.uint8))
    with pytest.raises(ValueError):
        pre(np.zeros((256, 256, 3), np.float32))


def test_autoencoder_rejects_indivisible_size():
    with pytest.raises(ValueError):
        AutoencoderPreprocessor(image_size=250)


def test_pca_flatten_roundtrip_and_determinism(cfg):
    pre = build_preprocessor("pca", cfg)
    assert isinstance(pre, PCAPreprocessor)
    ds = build_dataset(cfg, "pca", "train", "fake_rgb")
    a, b = ds[0]["image"], ds[0]["image"]
    assert torch.equal(a, b)
    flat = pre.flatten(a.unsqueeze(0))
    assert flat.shape == (1, 256 * 256)
    assert torch.equal(pre.unflatten(flat), a.unsqueeze(0))
    assert pre.describe()["patch_grid"] == (15, 15)
