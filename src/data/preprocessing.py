"""Shared image preprocessing helpers."""
import torch
import torch.nn.functional as F


def to_grayscale(images):
    """Convert BCHW RGB tensor to BCHW luminance."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected a BCHW RGB tensor")
    weights = images.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def resize_map(anomaly_map, size):
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(1)
    return F.interpolate(anomaly_map.float(), size=size, mode="bilinear", align_corners=False)


def normalize_map(anomaly_map, eps=1e-8):
    lo = anomaly_map.amin(dim=(-2, -1), keepdim=True)
    hi = anomaly_map.amax(dim=(-2, -1), keepdim=True)
    return (anomaly_map - lo) / (hi - lo + eps)
