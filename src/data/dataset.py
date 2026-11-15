"""Raw MVTec AD file loader with a deterministic normal-only validation split.

This module only indexes files and reads their stored pixel arrays. It does
not resize, change colour modes, scale pixel values, threshold masks, or
convert arrays to tensors. Those operations belong in preprocessing code.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """Load original images/masks for one category and split.

    Images are returned as NumPy arrays in their original stored mode,
    dimensions, and dtype. Masks are returned as original NumPy arrays;
    normal samples have mask=None. Labels and metadata are returned as
    simple Python values.

    Validation is held out from train/good only. The official test set is
    never used to construct the train/validation split.
    """

    def __init__(self, root, category, split="train",
                 val_fraction=0.10, seed=42):
        self.root = Path(root)
        self.category = category
        self.split = split
        base = self.root / category
        self.samples = []

        if split in {"train", "val"}:
            folder = base / "train" / "good"
            images = sorted(folder.glob("*.png"))
            if not images:
                raise FileNotFoundError(f"No training images found in {folder}")
            if not 0.0 <= val_fraction < 1.0:
                raise ValueError("val_fraction must be in [0, 1)")

            indices = np.arange(len(images))
            np.random.default_rng(seed).shuffle(indices)
            n_val = int(round(len(images) * val_fraction))
            if val_fraction > 0 and len(images) > 1:
                n_val = max(1, n_val)
            n_val = min(n_val, max(0, len(images) - 1))
            val_ids = set(indices[:n_val].tolist())
            chosen = [
                p for i, p in enumerate(images)
                if (i in val_ids) == (split == "val")
            ]
            self.samples = [(p, None, "good") for p in chosen]

        elif split == "test":
            test_root = base / "test"
            if not test_root.is_dir():
                raise FileNotFoundError(f"Missing test directory: {test_root}")
            for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
                defect = defect_dir.name
                for image in sorted(defect_dir.glob("*.png")):
                    mask = (
                        None if defect == "good"
                        else base / "ground_truth" / defect / f"{image.stem}_mask.png"
                    )
                    self.samples.append((image, mask, defect))
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        if not self.samples:
            raise RuntimeError(
                f"No samples for {category}/{split} under {self.root}. "
                "Check that root directly contains category folders."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, defect = self.samples[index]

        # Read and copy the original pixel array; do not transform it.
        with Image.open(image_path) as im:
            image = np.array(im)

        mask = None
        if mask_path is not None:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing ground-truth mask: {mask_path}")
            with Image.open(mask_path) as m:
                mask = np.array(m)

        return {
            "image": image,
            "mask": mask,
            "label": int(defect != "good"),
            "defect_type": defect,
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
        }


def available_categories(root):
    """List valid MVTec category directories under root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "train" / "good").is_dir()
    )
