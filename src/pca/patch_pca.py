"""Patch-wise PCA anomaly detector.

Overlapping patches are extracted from normal training images. PCA is fitted
on a deterministic sample of those patches to control memory use. At inference,
each patch is reconstructed and its mean squared pixel error is accumulated
back over the pixels it covers; overlapping contributions are averaged.

PCA linear algebra is implemented with NumPy (method of snapshots). PyTorch is
used only for convenient input conversion if a tensor is supplied.
"""
from __future__ import annotations

import numpy as np


def _to_numpy(images):
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 3:  # N,H,W -> N,1,H,W
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError("Expected images with shape (N,C,H,W) or (N,H,W).")
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN or Inf.")
    return x


class PatchPCA:
    """Local PCA reconstruction baseline.

    Args:
        patch_size: square patch width.
        stride: distance between adjacent patch origins.
        n_components: integer count or retained-variance target in (0,1].
        max_train_patches: maximum patches used to fit PCA. Sampling is
            deterministic and spread over the full patch collection.
        random_state: seed for reproducible patch subsampling.
    """

    def __init__(self, patch_size=32, stride=16, n_components=0.95,
                 max_train_patches=50000, random_state=42, eps=1e-12):
        if patch_size < 1 or stride < 1:
            raise ValueError("patch_size and stride must be positive.")
        if isinstance(n_components, float) and not (0.0 < n_components <= 1.0):
            raise ValueError("Float n_components must be in (0, 1].")
        if isinstance(n_components, int) and n_components < 1:
            raise ValueError("Integer n_components must be >= 1.")
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.n_components = n_components
        self.max_train_patches = int(max_train_patches)
        self.random_state = int(random_state)
        self.eps = float(eps)
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.image_shape_ = None
        self.grid_shape_ = None
        self.n_patches_seen_ = 0

    def _extract(self, images):
        """Return patches as (N, L, patch_dimension), row-major grid order."""
        x = _to_numpy(images)
        n, c, h, w = x.shape
        p, s = self.patch_size, self.stride
        if h < p or w < p:
            raise ValueError(f"Image {(h, w)} is smaller than patch size {p}.")
        if (h - p) % s or (w - p) % s:
            raise ValueError(
                "Patch grid does not tile cleanly: (image_size - patch_size) "
                "must be divisible by stride."
            )
        gh = (h - p) // s + 1
        gw = (w - p) // s + 1
        patches = np.empty((n, gh * gw, c * p * p), dtype=np.float32)
        idx = 0
        for top in range(0, h - p + 1, s):
            for left in range(0, w - p + 1, s):
                patches[:, idx, :] = x[:, :, top:top+p, left:left+p].reshape(n, -1)
                idx += 1
        return patches, (n, c, h, w), (gh, gw)

    def fit(self, images):
        """Fit on normal training images only."""
        patches, self.image_shape_, self.grid_shape_ = self._extract(images)
        flat = patches.reshape(-1, patches.shape[-1]).astype(np.float64)
        self.n_patches_seen_ = len(flat)
        if len(flat) < 2:
            raise ValueError("Patch PCA needs at least two training patches.")

        if self.max_train_patches > 0 and len(flat) > self.max_train_patches:
            # Evenly spaced deterministic subsampling; no validation/test data.
            chosen = np.linspace(0, len(flat) - 1, self.max_train_patches, dtype=np.int64)
            flat = flat[chosen]

        self.mean_ = flat.mean(axis=0)
        centered = flat - self.mean_
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        positive = eigvals > self.eps * max(float(eigvals[0]), 1.0)
        eigvals, eigvecs = eigvals[positive], eigvecs[:, positive]
        if eigvals.size == 0:
            raise ValueError("Training patches have no measurable variance.")

        components = (eigvecs.T @ centered) / np.sqrt(eigvals[:, None])
        components /= np.maximum(np.linalg.norm(components, axis=1, keepdims=True), self.eps)
        ratios = eigvals / eigvals.sum()
        if isinstance(self.n_components, float):
            k = int(np.searchsorted(np.cumsum(ratios), self.n_components) + 1)
        else:
            k = int(self.n_components)
        k = max(1, min(k, len(eigvals)))

        self.components_ = components[:k]
        self.eigenvalues_ = eigvals[:k]
        self.explained_variance_ratio_ = ratios[:k]
        return self

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("Call fit() on normal training images before anomaly_map().")

    def reconstruct_patches(self, patches):
        """Reconstruct flattened patch vectors, shape (N*L, patch_dimension)."""
        self._check_fitted()
        patches = np.asarray(patches, dtype=np.float64)
        scores = (patches - self.mean_) @ self.components_.T
        return scores @ self.components_ + self.mean_

    def anomaly_map(self, images):
        """Return overlapping-patch reconstruction error maps, shape (N,H,W)."""
        self._check_fitted()
        patches, shape, grid = self._extract(images)
        if shape[1:] != self.image_shape_[1:]:
            raise ValueError(
                f"Expected image shape {self.image_shape_[1:]}, got {shape[1:]}."
            )
        n, c, h, w = shape
        gh, gw = grid
        p, s = self.patch_size, self.stride
        flat = patches.reshape(-1, patches.shape[-1]).astype(np.float64)
        recon = self.reconstruct_patches(flat)
        patch_error = np.mean((flat - recon) ** 2, axis=1).reshape(n, gh, gw)

        # Overlap-add: each patch contributes its error to every covered pixel.
        maps = np.zeros((n, h, w), dtype=np.float64)
        counts = np.zeros((h, w), dtype=np.float64)
        for row, top in enumerate(range(0, h - p + 1, s)):
            for col, left in enumerate(range(0, w - p + 1, s)):
                maps[:, top:top+p, left:left+p] += patch_error[:, row, col, None, None]
                counts[top:top+p, left:left+p] += 1.0
        maps /= np.maximum(counts[None, :, :], 1.0)
        return maps.astype(np.float32)

    def image_score(self, images, aggregation="max"):
        maps = self.anomaly_map(images)
        flat = maps.reshape(len(maps), -1)
        if aggregation == "max":
            return flat.max(axis=1)
        if aggregation == "mean":
            return flat.mean(axis=1)
        raise ValueError("aggregation must be 'max' or 'mean'.")

    def save(self, path):
        self._check_fitted()
        np.savez_compressed(
            path, mean=self.mean_, components=self.components_,
            eigenvalues=self.eigenvalues_,
            explained_variance_ratio=self.explained_variance_ratio_,
            image_shape=np.asarray(self.image_shape_, dtype=np.int64),
            grid_shape=np.asarray(self.grid_shape_, dtype=np.int64),
            patch_size=self.patch_size, stride=self.stride,
        )
