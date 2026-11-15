"""MVTec AD loader with a deterministic normal-only validation split."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """Load one category's train, validation, or official test split.

    Validation is held out from train/good only. Test data is never used
    to construct the train/validation split.
    """
    def __init__(self, root, category, split="train", image_size=256,
                 val_fraction=0.10, seed=42):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
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
            chosen = [p for i, p in enumerate(images)
                      if (i in val_ids) == (split == "val")]
            self.samples = [(p, None, "good") for p in chosen]

        elif split == "test":
            test_root = base / "test"
            if not test_root.is_dir():
                raise FileNotFoundError(f"Missing test directory: {test_root}")
            for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
                defect = defect_dir.name
                for image in sorted(defect_dir.glob("*.png")):
                    mask = (None if defect == "good" else
                            base / "ground_truth" / defect / f"{image.stem}_mask.png")
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
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image = np.asarray(im, dtype=np.float32) / 255.0

        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if mask_path is not None:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing ground-truth mask: {mask_path}")
            with Image.open(mask_path) as m:
                m = m.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST)
                mask = (np.asarray(m) > 0).astype(np.uint8)

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "label": torch.tensor(float(defect != "good")),
            "defect_type": defect,
            "image_path": str(image_path),
        }


def available_categories(root):
    """List valid MVTec category directories under root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "train" / "good").is_dir())
