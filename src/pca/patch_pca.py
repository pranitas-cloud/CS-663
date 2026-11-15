"""Patch-wise PCA model for local reconstruction error."""
import numpy as np
import torch
import torch.nn.functional as F


class PatchPCA:
    def __init__(self, patch_size=32, stride=16, n_components=0.95):
        self.patch_size, self.stride, self.n_components = patch_size, stride, n_components
        self.mean_, self.components_ = None, None
        self.image_shape_ = None

    def _patches(self, x):
        # x: N,C,H,W
        n, c, h, w = x.shape
        patches = F.unfold(torch.as_tensor(x).float(), self.patch_size, stride=self.stride)
        return patches.transpose(1, 2).reshape(-1, c * self.patch_size ** 2), (n, c, h, w)

    def fit(self, images):
        patches, self.image_shape_ = self._patches(images)
        x = patches.numpy().astype(np.float64)
        self.mean_ = x.mean(0)
        centered = x - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            k = np.searchsorted(np.cumsum(s*s) / np.sum(s*s), self.n_components) + 1
        else:
            k = int(self.n_components)
        self.components_ = vt[:max(1, min(k, len(vt)))]
        return self

    def anomaly_map(self, images):
        x = torch.as_tensor(images).float()
        n, c, h, w = x.shape
        patches = F.unfold(x, self.patch_size, stride=self.stride).transpose(1, 2)
        a = patches.numpy().reshape(-1, patches.shape[-1]).astype(np.float64)
        z = (a - self.mean_) @ self.components_.T
        rec = z @ self.components_ + self.mean_
        err = torch.from_numpy(((a - rec) ** 2).mean(1).astype(np.float32)).view(n, -1, 1)
        grid_h = (h - self.patch_size) // self.stride + 1
        grid_w = (w - self.patch_size) // self.stride + 1
        maps = err.transpose(1, 2).reshape(n, 1, grid_h, grid_w)
        maps = F.interpolate(maps, size=(h, w), mode="bilinear", align_corners=False)
        return maps
