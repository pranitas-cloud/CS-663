"""Global PCA reconstruction baseline, implemented with NumPy."""
from pathlib import Path
import argparse
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import MVTecDataset
from src.data.preprocessing import to_grayscale


class GlobalPCA:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, images):
        x = np.asarray(images, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(len(x), -1)
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            energy = np.cumsum(s ** 2) / np.sum(s ** 2)
            k = int(np.searchsorted(energy, self.n_components) + 1)
        else:
            k = int(self.n_components)
        self.components_ = vt[:max(1, min(k, len(vt)))]
        return self

    def reconstruct(self, images):
        x = np.asarray(images, dtype=np.float64)
        shape = x.shape
        x = x.reshape(len(x), -1)
        z = (x - self.mean_) @ self.components_.T
        return (z @ self.components_ + self.mean_).reshape(shape).astype(np.float32)

    def score_map(self, images):
        x = np.asarray(images, dtype=np.float32)
        recon = self.reconstruct(x)
        return np.mean((x - recon) ** 2, axis=1, keepdims=True) if x.ndim == 4 else (x - recon) ** 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    p.add_argument("--category", default="bottle")
    p.add_argument("--components", type=float, default=0.95)
    args = p.parse_args()
    ds = MVTecDataset(args.data_root, args.category, "train", 64)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    x = np.concatenate([to_grayscale(b["image"]).numpy().reshape(len(b["image"]), -1) for b in loader])
    model = GlobalPCA(args.components).fit(x)
    out = Path("experiments/pca"); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.category}_global_pca.npz", mean=model.mean_, components=model.components_)
    print(f"Fitted {len(model.components_)} components on {len(x)} normal images.")


if __name__ == "__main__":
    main()
