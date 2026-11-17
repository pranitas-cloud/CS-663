from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", val_fraction=0.10, seed=42):
        self.root = Path(root)
        self.category = category
        self.split = split
        base = self.root / category
        self.samples = []

        if split in ("train", "val"):
            folder = base / "train" / "good"
            images = sorted(folder.glob("*.png"))
            if not images:
                raise FileNotFoundError(f"No training images found in {folder}")
            if not 0 <= val_fraction < 1:
                raise ValueError("val_fraction must be in [0, 1).")
            indices = np.arange(len(images))
            np.random.default_rng(seed).shuffle(indices)
            n_val = int(round(len(images) * val_fraction))
            if val_fraction > 0 and len(images) > 1:
                n_val = max(1, n_val)
            n_val = min(n_val, len(images) - 1)
            validation_ids = set(indices[:n_val].tolist())
            selected = [
                path for index, path in enumerate(images)
                if (index in validation_ids) == (split == "val")
            ]
            self.samples = [(path, None, "good") for path in selected]
        elif split == "test":
            test_root = base / "test"
            if not test_root.is_dir():
                raise FileNotFoundError(f"Missing test directory: {test_root}")
            for defect_folder in sorted(path for path in test_root.iterdir() if path.is_dir()):
                defect = defect_folder.name
                for path in sorted(defect_folder.glob("*.png")):
                    mask = None if defect == "good" else base / "ground_truth" / defect / f"{path.stem}_mask.png"
                    self.samples.append((path, mask, defect))
        else:
            raise ValueError("split must be train, val or test.")

        if not self.samples:
            raise RuntimeError(f"No samples found for {category}/{split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, defect = self.samples[index]
        with Image.open(image_path) as image_file:
            image = np.array(image_file)
        mask = None
        if mask_path is not None:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            with Image.open(mask_path) as mask_file:
                mask = np.array(mask_file)
        return {
            "image": image,
            "mask": mask,
            "label": int(defect != "good"),
            "defect_type": defect,
            "image_path": str(image_path),
        }


def available_categories(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "train" / "good").is_dir())
