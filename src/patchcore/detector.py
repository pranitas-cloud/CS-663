"""PatchCore-style feature-memory training and inference pipeline."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.pipelines import build_dataset
from src.patchcore.features import FeatureExtractor
from src.patchcore.memory_bank import MemoryBank


class PatchCoreDetector:
    def __init__(self, max_samples=10000, pretrained=True, device=None):
        self.extractor = FeatureExtractor(pretrained=pretrained, device=device)
        self.bank = MemoryBank(max_samples)
        self.device = self.extractor.device

    @torch.no_grad()
    def fit(self, loader):
        batches = []
        for batch in loader:
            batches.append(self.extractor(batch["image"]))
        self.bank.fit(batches)
        return self

    @torch.no_grad()
    def predict(self, images):
        features = self.extractor(images)
        patch_scores = self.bank.query(features)
        batch_size, patch_count = patch_scores.shape
        side = int(patch_count ** 0.5)
        if side * side != patch_count:
            raise ValueError("Feature grid must be square for current map reconstruction.")
        anomaly_map = patch_scores.reshape(batch_size, 1, side, side)
        anomaly_map = F.interpolate(
            anomaly_map,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        image_scores = patch_scores.max(dim=1).values
        return image_scores, anomaly_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", default="experiments/patchcore")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "seed": 42,
        "data": {
            "root": args.data_root,
            "category": args.category,
            "val_fraction": 0.10,
            "image_size": args.image_size,
            "mask_resize": "nearest",
        },
        "patchcore": {"image_size": args.image_size},
    }
    train_dataset = build_dataset(config, "patchcore", "train", args.category)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    detector = PatchCoreDetector(
        max_samples=args.max_samples,
        pretrained=not args.no_pretrained,
    ).fit(train_loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.category}_memory.pt"
    torch.save(
        {
            "features": detector.bank.features,
            "category": args.category,
            "image_size": args.image_size,
            "max_samples": args.max_samples,
            "pretrained": not args.no_pretrained,
        },
        output_path,
    )
    print(f"Saved memory bank with {len(detector.bank.features)} patch features: {output_path}")


if __name__ == "__main__":
    main()
