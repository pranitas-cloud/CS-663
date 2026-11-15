"""Method-specific preprocessing pipelines on top of the raw ``MVTecDataset``.

Each pipeline maps (raw image array, raw mask array or None) to tensors. None
of them fit any statistic on data, so nothing can leak from the test set.

Choices common to all three methods (justified by the dataset audit):
  * square resize only, no crop: every image is square, and a centre crop
    (as in the PatchCore reference code) can discard border defects and
    encodes a prior on where defects lie;
  * anti-aliased bilinear resize for images, since the source is 3-4x larger
    than the working resolution;
  * masks are binarised (> 0) and resized with nearest neighbour;
  * no flip/rotation augmentation: metal_nut has a ``flip`` defect type and
    transistor has misplaced parts, so a flip can turn a defect into "normal".
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.dataset import MVTecDataset
from src.data.preprocessing import (
    check_square,
    extract_patches,
    fold_patches,
    image_to_tensor,
    mask_to_tensor,
    normalize_imagenet,
    patch_grid,
    resize_image,
)


class BasePreprocessor:
    """Common interface: ``pre(image, mask) -> (image_tensor, mask_tensor)``."""

    method = "base"
    grayscale = False

    def __init__(self, image_size=256, mask_mode="nearest"):
        self.image_size = int(image_size)
        self.mask_mode = mask_mode

    @property
    def channels(self):
        return 1 if self.grayscale else 3

    def _base(self, image):
        """Raw array -> C x S x S float tensor in [0, 1] (no normalisation)."""
        check_square(image.shape)
        t = image_to_tensor(image, grayscale=self.grayscale)
        return resize_image(t, self.image_size)

    def image(self, image):
        return self._base(image)

    def mask(self, mask):
        return mask_to_tensor(mask, self.image_size, mode=self.mask_mode)

    def __call__(self, image, mask=None):
        return self.image(image), self.mask(mask)

    def describe(self):
        """Settings to store next to every result (methodology.md requirement)."""
        return {
            "method": self.method,
            "image_size": self.image_size,
            "channels": self.channels,
            "resize": "bilinear, antialias=True, square, no crop",
            "mask_resize": self.mask_mode,
        }


class PCAPreprocessor(BasePreprocessor):
    """Grayscale [0, 1] images for global and patch-wise PCA.

    Mean-centring is NOT done here: the PCA fit subtracts the mean of the
    training set (and only the training set), then applies it to val/test.
    Single-channel source images are used unchanged; RGB ones become luminance.
    """

    method = "pca"

    def __init__(self, image_size=256, grayscale=True, patch_size=32, stride=16,
                 mask_mode="nearest"):
        super().__init__(image_size, mask_mode)
        self.grayscale = bool(grayscale)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        patch_grid(self.image_size, self.patch_size, self.stride)  # validate tiling

    # global PCA -----------------------------------------------------------
    def flatten(self, images):
        """B x C x H x W -> B x (C*H*W), the data matrix for global PCA."""
        return images.reshape(images.shape[0], -1)

    def unflatten(self, flat):
        """B x (C*H*W) -> B x C x H x W."""
        return flat.reshape(flat.shape[0], self.channels, self.image_size, self.image_size)

    # patch PCA ------------------------------------------------------------
    def patches(self, images):
        """B x C x H x W -> B x L x (C*p*p)."""
        return extract_patches(images, self.patch_size, self.stride)

    def fold(self, patches):
        """B x L x (C*p*p) -> B x C x H x W, overlaps averaged."""
        return fold_patches(patches, self.channels, self.image_size,
                            self.patch_size, self.stride)

    def describe(self):
        d = super().describe()
        d.update(scaling="[0,1]", centring="done in PCA fit on train only",
                 patch_size=self.patch_size, stride=self.stride,
                 patch_grid=patch_grid(self.image_size, self.patch_size, self.stride))
        return d


class AutoencoderPreprocessor(BasePreprocessor):
    """RGB [0, 1] images for the convolutional autoencoder.

    Colour is kept because several categories have colour defects (carpet,
    leather, metal_nut, pill, ...). Single-channel categories (grid, screw,
    zipper) are replicated to three channels so one network shape serves all.
    Values stay in [0, 1] so the decoder can end in a sigmoid.
    """

    method = "autoencoder"

    def __init__(self, image_size=256, grayscale=False, divisible_by=32,
                 mask_mode="nearest"):
        super().__init__(image_size, mask_mode)
        self.grayscale = bool(grayscale)
        if self.image_size % divisible_by != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by {divisible_by} "
                "(2**number of stride-2 stages) so decoder output matches the input."
            )
        self.divisible_by = divisible_by

    def describe(self):
        d = super().describe()
        d.update(scaling="[0,1] (no mean/std)", augmentation="none",
                 divisible_by=self.divisible_by)
        return d


class PatchCorePreprocessor(BasePreprocessor):
    """RGB + ImageNet standardisation for the pretrained Wide-ResNet-50-2.

    Resize straight to ``image_size`` (default 256) with no centre crop. At 256
    px, layer2 (stride 8) is 32x32 and layer3 (stride 16) is 16x16.
    Grayscale sources are replicated to three channels, as the backbone needs.
    """

    method = "patchcore"
    grayscale = False

    def __init__(self, image_size=256, mask_mode="nearest"):
        super().__init__(image_size, mask_mode)

    def image(self, image):
        return normalize_imagenet(self._base(image))

    def describe(self):
        d = super().describe()
        d.update(scaling="[0,1] then ImageNet mean/std", center_crop=None)
        return d


# --------------------------------------------------------------------------
# Config -> preprocessor -> dataset
# --------------------------------------------------------------------------
def build_preprocessor(method, cfg):
    """Create the preprocessor for ``method`` from a loaded config dict.

    Per-method keys override ``data.image_size``.
    """
    data = cfg.get("data", {})
    mask_mode = data.get("mask_resize", "nearest")
    section = cfg.get(method, {})
    size = section.get("image_size", data.get("image_size", 256))

    if method == "pca":
        return PCAPreprocessor(
            image_size=size,
            grayscale=section.get("grayscale", True),
            patch_size=section.get("patch_size", 32),
            stride=section.get("stride", 16),
            mask_mode=mask_mode,
        )
    if method == "autoencoder":
        return AutoencoderPreprocessor(
            image_size=size,
            grayscale=section.get("grayscale", False),
            divisible_by=section.get("divisible_by", 32),
            mask_mode=mask_mode,
        )
    if method == "patchcore":
        return PatchCorePreprocessor(image_size=size, mask_mode=mask_mode)
    raise ValueError(f"Unknown method '{method}'. Use pca, autoencoder or patchcore.")


class PreparedDataset(Dataset):
    """Wrap a raw ``MVTecDataset`` and apply a preprocessor.

    Every item has the same keys and tensor types, so the default DataLoader
    collate works: normal samples carry an all-zero mask.
    ``orig_size`` keeps the source (H, W), needed to map scores back to the
    original resolution or to measure defect sizes in source pixels.
    """

    def __init__(self, raw, preprocessor):
        self.raw = raw
        self.pre = preprocessor
        self.samples = raw.samples  # keeps notebook code like train_ds.samples working

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        s = self.raw[index]
        image, mask = self.pre(s["image"], s["mask"])
        h, w = s["image"].shape[:2]
        return {
            "image": image,
            "mask": mask,
            "label": s["label"],
            "defect_type": s["defect_type"],
            "image_path": s["image_path"],
            "orig_size": torch.tensor([h, w]),
        }


def build_dataset(cfg, method, split, category=None):
    """Raw loader + method preprocessing for one category and split."""
    data = cfg["data"]
    raw = MVTecDataset(
        Path(data["root"]),
        category or data["category"],
        split=split,
        val_fraction=data.get("val_fraction", 0.10),
        seed=cfg.get("seed", 42),
    )
    return PreparedDataset(raw, build_preprocessor(method, cfg))


def stack_images(dataset):
    """Stack all images of a (train) dataset into one N x C x H x W tensor.

    Use this to fit PCA or to build the PatchCore memory bank. Call it on the
    train split only; ``val`` is for thresholds/early stopping, ``test`` is
    for final numbers.
    """
    return torch.stack([dataset[i]["image"] for i in range(len(dataset))])


def load_config(path="configs/default.yaml"):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
