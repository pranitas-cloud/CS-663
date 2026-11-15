"""Shared image preprocessing primitives.

The raw loader (``src/data/dataset.py``) returns original NumPy arrays: 2-D
(``L``) or H x W x 3 (``RGB``) uint8 images, and 2-D uint8 masks with values
{0, 255}. Everything below turns those into tensors. Method-specific choices
(grayscale, normalisation, patching) live in ``src/data/pipelines.py``.

Dataset facts these helpers rely on (from the exploration/audit notebooks):
  * every image and mask is square (700, 800, 840, 900, 1000 or 1024 px), so a
    plain square resize never distorts aspect ratio and no crop is needed;
  * 1,213 of 5,354 images are stored single-channel, the rest are RGB;
  * masks are single-channel with values {0, 255}, same size as their image.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Existing helpers (unchanged behaviour)
# --------------------------------------------------------------------------
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
    """Per-image min-max scaling, for visualisation only.

    Do NOT apply this before computing image AUROC or pixel AUROC: it removes
    the absolute score scale that makes scores comparable across images.
    """
    lo = anomaly_map.amin(dim=(-2, -1), keepdim=True)
    hi = anomaly_map.amax(dim=(-2, -1), keepdim=True)
    return (anomaly_map - lo) / (hi - lo + eps)


# --------------------------------------------------------------------------
# Raw array -> tensor
# --------------------------------------------------------------------------
def check_square(shape):
    """Raise if an (H, W, ...) shape is not square.

    The whole dataset is square. A square resize of a non-square image would
    silently distort it, so fail loudly instead.
    """
    h, w = int(shape[0]), int(shape[1])
    if h != w:
        raise ValueError(
            f"Expected square images (all MVTec AD images are), got {h}x{w}."
        )
    return h


def _as_hwc_uint8(image):
    """Return a contiguous H x W x C uint8 array with C in {1, 3}."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {arr.dtype}.")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported image shape {arr.shape}.")
    if arr.shape[2] == 4:  # drop alpha if it ever appears
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def image_to_tensor(image, grayscale=False):
    """Raw uint8 array -> float tensor C x H x W scaled to [0, 1].

    Single-channel (``L``) images are used as-is when ``grayscale`` is True and
    replicated to three identical channels otherwise, so every category yields
    the same channel count regardless of how the file is stored.
    """
    arr = _as_hwc_uint8(image)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    if grayscale:
        if t.shape[0] == 1:
            return t
        return to_grayscale(t.unsqueeze(0)).squeeze(0)
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1).contiguous()
    return t


def resize_image(image, size):
    """Resize C x H x W (or B x C x H x W) to ``size`` x ``size``.

    Bilinear with anti-aliasing. The source is 700-1024 px and the target is
    typically 256 px, so a 3-4x reduction; without anti-aliasing, thin
    scratches and fine textures alias into spurious patterns.
    """
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    size = (size, size) if isinstance(size, int) else tuple(size)
    if tuple(image.shape[-2:]) != size:
        image = F.interpolate(
            image, size=size, mode="bilinear", align_corners=False, antialias=True
        ).clamp_(0.0, 1.0)
    return image.squeeze(0) if squeeze else image


def normalize_imagenet(image):
    """Standardise a [0, 1] RGB tensor with ImageNet statistics (C x H x W or B x C x H x W)."""
    mean = image.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return (image - mean) / std


def denormalize_imagenet(image):
    """Inverse of ``normalize_imagenet`` (for plotting)."""
    mean = image.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return image * std + mean


def mask_to_tensor(mask, size, mode="nearest"):
    """Raw mask (or None for a normal sample) -> float tensor 1 x size x size in {0, 1}.

    * ``None`` becomes an all-zero mask, so normal and defective samples batch
      together and evaluation can treat every image uniformly.
    * Any non-zero pixel is foreground (masks hold {0, 255}).
    * ``mode="nearest"`` is the project default (see methodology.md).
      ``mode="max"`` marks a target pixel as defect if ANY source pixel under
      it is defect, which keeps tiny defects alive when downsampling.
    """
    if mask is None:
        return torch.zeros(1, size, size)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.ndim != 2:
        raise ValueError(f"Unsupported mask shape {np.asarray(mask).shape}.")
    check_square(m.shape)
    t = torch.from_numpy(np.ascontiguousarray(m > 0)).float()[None, None]
    if t.shape[-1] != size:
        if mode == "nearest":
            t = F.interpolate(t, size=(size, size), mode="nearest")
        elif mode == "max":
            if size > t.shape[-1]:
                raise ValueError("mode='max' only supports downsampling.")
            t = F.adaptive_max_pool2d(t, (size, size))
        else:
            raise ValueError(f"Unknown mask mode '{mode}'.")
    return t.squeeze(0)


# --------------------------------------------------------------------------
# Patch extraction (used by patch-wise PCA)
# --------------------------------------------------------------------------
def patch_grid(image_size, patch_size, stride):
    """Return (rows, cols) of the patch grid and verify full coverage.

    With ``(image_size - patch_size) % stride != 0`` the last strip of pixels
    is never covered, which would leave NaNs when folding scores back.
    """
    if patch_size > image_size or stride < 1:
        raise ValueError("Need 1 <= stride and patch_size <= image_size.")
    if (image_size - patch_size) % stride != 0:
        raise ValueError(
            f"Patches do not tile the image: ({image_size} - {patch_size}) "
            f"is not divisible by stride {stride}."
        )
    n = (image_size - patch_size) // stride + 1
    return n, n


def extract_patches(images, patch_size, stride):
    """B x C x H x W -> B x L x (C * p * p) overlapping patches, row-major order."""
    if images.ndim != 4:
        raise ValueError("Expected a BCHW tensor.")
    patch_grid(images.shape[-1], patch_size, stride)
    cols = F.unfold(images, kernel_size=patch_size, stride=stride)  # B x (C*p*p) x L
    return cols.transpose(1, 2).contiguous()


def fold_patches(patches, channels, image_size, patch_size, stride):
    """Inverse of ``extract_patches``; overlapping values are averaged.

    ``patches`` is B x L x (channels * p * p). Returns B x channels x H x W.
    Used to paint per-patch reconstruction residuals back into a score map.
    """
    patch_grid(image_size, patch_size, stride)
    out_size = (image_size, image_size)
    cols = patches.transpose(1, 2)
    summed = F.fold(cols, output_size=out_size, kernel_size=patch_size, stride=stride)
    counts = F.fold(
        torch.ones_like(cols), output_size=out_size, kernel_size=patch_size, stride=stride
    )
    return summed / counts
