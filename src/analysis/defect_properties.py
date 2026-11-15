"""Compute interpretable geometric/intensity descriptors from binary masks."""
import numpy as np
from scipy import ndimage


def describe_defect(mask, image=None):
    mask = np.asarray(mask).squeeze().astype(bool)
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return {"area_pixels": 0, "area_fraction": 0.0, "bbox_aspect_ratio": 0.0,
                "centroid_x": None, "centroid_y": None, "components": 0,
                "perimeter_pixels": 0, "contrast": None}
    labels, count = ndimage.label(mask)
    ys, xs = np.where(mask)
    bbox_w, bbox_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    contrast = None
    if image is not None:
        im = np.asarray(image)
        if im.ndim == 3 and im.shape[0] in (1, 3): im = im.transpose(1, 2, 0)
        if im.ndim == 3: im = im.mean(axis=2)
        contrast = float(abs(im[mask].mean() - im[~mask].mean())) if (~mask).any() else 0.0
    eroded = ndimage.binary_erosion(mask)
    perimeter = int((mask ^ eroded).sum())
    return {"area_pixels": area, "area_fraction": float(area/(h*w)),
            "bbox_aspect_ratio": float(bbox_w/max(bbox_h, 1)),
            "centroid_x": float(xs.mean()/max(w-1, 1)),
            "centroid_y": float(ys.mean()/max(h-1, 1)),
            "components": int(count), "perimeter_pixels": perimeter,
            "contrast": contrast}
