"""MVTec AD dataset loader."""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """Load train/good or test images for one MVTec AD category."""
    def __init__(self, root, category, split="train", image_size=256):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.samples = []
        base = self.root / category
        if split == "train":
            folder = base / "train" / "good"
            self.samples = [(p, None, "good") for p in sorted(folder.glob("*.png"))]
        elif split == "test":
            test_root = base / "test"
            for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
                defect = defect_dir.name
                for image in sorted(defect_dir.glob("*.png")):
                    mask = None if defect == "good" else base / "ground_truth" / defect / f"{image.stem}_mask.png"
                    self.samples.append((image, mask, defect))
        else:
            raise ValueError("split must be 'train' or 'test'")
        if not self.samples:
            raise FileNotFoundError(f"No images found for {category}/{split} under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, defect = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if mask_path is not None:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing ground-truth mask: {mask_path}")
            m = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask = (np.asarray(m) > 0).astype(np.uint8)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "label": torch.tensor(float(defect != "good")),
            "defect_type": defect,
            "image_path": str(image_path),
        }


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()
    ds = MVTecDataset(args.data_root, args.category, "test")
    batch = next(iter(DataLoader(ds, batch_size=4)))
    print(f"{len(ds)} test images; batch shape {tuple(batch['image'].shape)}")
