"""Shared interface for all anomaly detectors."""
from abc import ABC, abstractmethod
import numpy as np


class AnomalyMethod(ABC):
    """All methods fit normal-only training images and return N×H×W maps.

    Input images are float32 arrays/tensors in NCHW format, scaled to [0, 1].
    """
    @abstractmethod
    def fit(self, train_images):
        """Fit on defect-free training images only."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, images) -> np.ndarray:
        """Return anomaly maps with shape (N, H, W)."""
        raise NotImplementedError


def ensure_anomaly_maps(maps, batch_size, height, width):
    maps = np.asarray(maps)
    if maps.shape != (batch_size, height, width):
        raise ValueError(
            f"Detector must return {(batch_size, height, width)}, got {maps.shape}"
        )
    if not np.isfinite(maps).all():
        raise ValueError("Anomaly maps contain NaN or infinite values")
    return maps.astype(np.float32, copy=False)
