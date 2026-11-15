"""PatchCore-style training and inference pipeline."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import MVTecDataset
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
        feats = self.extractor(images)
        patch_scores = self.bank.query(feats)
        b, n = patch_scores.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError("Feature grid must be square for current map reconstruction.")
        amap = patch_scores.reshape(b, 1, side, side)
        amap = F.interpolate(amap, size=images.shape[-2:], mode="bilinear", align_corners=False)
        image_scores = patch_scores.max(dim=1).values
        return image_scores, amap

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    p.add_argument("--category", default="bottle")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    train = DataLoader(MVTecDataset(args.data_root, args.category, "train"), batch_size=args.batch_size)
    detector = PatchCoreDetector(args.max_samples).fit(train)
    out = Path("experiments/patchcore"); out.mkdir(parents=True, exist_ok=True)
    torch.save(detector.bank.features, out / f"{args.category}_memory.pt")
    print(f"Saved memory bank with {len(detector.bank.features)} patch features.")

if __name__ == "__main__":
    main()
