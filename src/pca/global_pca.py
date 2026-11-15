"""Global image PCA anomaly detector.

PCA is fitted on normal training images only. The implementation uses the
method of snapshots: eigendecompose the N x N Gram matrix X X^T, then map
the eigenvectors back into pixel space. This is useful here because the
number of pixels (D) is much larger than the number of training images (N).

Inputs may be NumPy arrays or torch tensors:
    fit:        (N,H,W), (N,C,H,W), or (N,D)
    anomaly_map:(N,H,W), (N,C,H,W), or (N,D)

No sklearn PCA is used.
"""
from __future__ import annotations

import numpy as np


def _to_numpy(images):
    """Convert an array/tensor to a finite float64 NumPy batch."""
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    x = np.asarray(images, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("Expected a batch of images with at least 2 dimensions.")
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN or Inf.")
    return x


class GlobalPCA:
    """Global PCA reconstruction baseline for grayscale or multi-channel images.

    Args:
        n_components: integer component count, or float variance target in
            (0,1), e.g. 0.95 for 95% retained training variance.
        eps: numerical tolerance for near-zero eigenvalues.
    """

    def __init__(self, n_components=0.95, eps=1e-12):
        if isinstance(n_components, float) and not (0.0 < n_components <= 1.0):
            raise ValueError("Float n_components must be in (0, 1].")
        if isinstance(n_components, int) and n_components < 1:
            raise ValueError("Integer n_components must be >= 1.")
        self.n_components = n_components
        self.eps = float(eps)
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None
        self.image_shape_ = None

    def fit(self, images):
        """Fit PCA on a batch containing normal training images only."""
        x = _to_numpy(images)
        self.image_shape_ = tuple(x.shape[1:])
        x = x.reshape(x.shape[0], -1)
        n_samples, n_features = x.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 training images.")

        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        self.n_features_in_ = n_features

        # Method of snapshots: (N,N), rather than the much larger (D,D)
        # covariance matrix. Eigenvectors are mapped back to pixel space.
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        positive = eigvals > self.eps * max(float(eigvals[0]), 1.0)
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]
        if eigvals.size == 0:
            raise ValueError("Training images have no measurable variance.")

        # Each row is one unit-length principal direction in pixel space.
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
            raise RuntimeError("Call fit() on normal training images before reconstruction.")

    def transform(self, images):
        """Project images onto the retained principal components."""
        self._check_fitted()
        x = _to_numpy(images)
        if x.shape[1:] != self.image_shape_:
            raise ValueError(f"Expected image shape {self.image_shape_}, got {x.shape[1:]}.")
        flat = x.reshape(x.shape[0], -1)
        return (flat - self.mean_) @ self.components_.T

    def inverse_transform(self, scores):
        """Reconstruct flattened images from PCA scores."""
        self._check_fitted()
        scores = _to_numpy(scores)
        if scores.ndim != 2 or scores.shape[1] != self.components_.shape[0]:
            raise ValueError("scores must have shape (N, n_components).")
        return scores @ self.components_ + self.mean_

    def reconstruct(self, images):
        """Return reconstructions with the same batch/image shape as input."""
        x = _to_numpy(images)
        if self.image_shape_ is None:
            raise RuntimeError("Call fit() before reconstruct().")
        if x.shape[1:] != self.image_shape_:
            raise ValueError(f"Expected image shape {self.image_shape_}, got {x.shape[1:]}.")
        return self.inverse_transform(self.transform(x)).reshape(x.shape).astype(np.float32)

    def anomaly_map(self, images, reduction="mean"):
        """Pixel-wise absolute reconstruction error, shape (N,H,W).

        For multi-channel input, channel errors are reduced at each pixel.
        This is an error map, not a probability map.
        """
        x = _to_numpy(images)
        recon = self.reconstruct(x).astype(np.float64)
        error = np.abs(x - recon)
        if len(self.image_shape_) == 3:  # C,H,W
            if reduction == "mean":
                error = error.mean(axis=1)
            elif reduction == "max":
                error = error.max(axis=1)
            else:
                raise ValueError("reduction must be 'mean' or 'max'.")
        return error.astype(np.float32)

    def image_score(self, images, aggregation="max"):
        """Convert each anomaly map to one image-level anomaly score."""
        maps = self.anomaly_map(images)
        if aggregation == "max":
            return maps.reshape(len(maps), -1).max(axis=1)
        if aggregation == "mean":
            return maps.reshape(len(maps), -1).mean(axis=1)
        raise ValueError("aggregation must be 'max' or 'mean'.")

    def save(self, path):
        """Save fitted parameters to an NPZ file."""
        self._check_fitted()
        np.savez_compressed(
            path, mean=self.mean_, components=self.components_,
            eigenvalues=self.eigenvalues_,
            explained_variance_ratio=self.explained_variance_ratio_,
            image_shape=np.asarray(self.image_shape_, dtype=np.int64),
        )

    @classmethod
    def load(cls, path):
        """Load a model saved with save()."""
        data = np.load(path)
        model = cls(n_components=int(len(data["components"])))
        model.mean_ = data["mean"]
        model.components_ = data["components"]
        model.eigenvalues_ = data["eigenvalues"]
        model.explained_variance_ratio_ = data["explained_variance_ratio"]
        model.image_shape_ = tuple(data["image_shape"].tolist())
        model.n_features_in_ = int(model.mean_.size)
        return model
