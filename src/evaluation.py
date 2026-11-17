import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import label as connected_components
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def image_score(anomaly_map, mode="top_1pct"):
    values = np.asarray(anomaly_map).reshape(-1)
    if mode == "max":
        return float(values.max())
    if mode == "top_1pct":
        count = max(1, int(np.ceil(values.size * 0.01)))
        return float(np.partition(values, -count)[-count:].mean())
    raise ValueError("mode must be max or top_1pct.")


def compute_aupro(masks, maps, max_fpr=0.30, n_thresholds=128):
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float64)
    if masks.shape != maps.shape or masks.ndim != 3:
        raise ValueError("Expected matching (N,H,W) masks and maps.")
    if not 0 < max_fpr <= 1 or n_thresholds < 2:
        raise ValueError("Invalid max_fpr or n_thresholds.")
    if not np.isfinite(maps).all():
        raise ValueError("Anomaly maps contain NaN or infinity.")

    labelled_masks = []
    region_count = 0
    for mask in masks:
        labels, count = connected_components(mask)
        labelled_masks.append(labels)
        region_count += count
    negatives = int((~masks).sum())
    if region_count == 0 or negatives == 0:
        return float("nan")

    thresholds = np.unique(np.concatenate((
        [np.nextafter(maps.max(), np.inf)],
        np.quantile(maps, np.linspace(1.0, 0.0, n_thresholds))
    )))[::-1]
    fprs = [0.0]
    pros = [0.0]

    for threshold in thresholds:
        false_positives = 0
        overlaps = []
        for index, labels in enumerate(labelled_masks):
            prediction = maps[index] >= threshold
            false_positives += np.logical_and(prediction, labels == 0).sum()
            if labels.max():
                predicted = np.bincount(labels[prediction], minlength=labels.max() + 1)[1:]
                areas = np.bincount(labels.ravel())[1:]
                overlaps.extend((predicted / areas).tolist())
        fpr = false_positives / negatives
        fprs.append(float(fpr))
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)
        if fpr >= max_fpr:
            break

    x = np.asarray(fprs)
    y = np.asarray(pros)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique_x = np.unique(x)
    unique_y = np.asarray([y[x == value].max() for value in unique_x])
    if unique_x[-1] < max_fpr:
        raise RuntimeError("AUPRO thresholds did not reach the FPR boundary.")
    endpoint = float(np.interp(max_fpr, unique_x, unique_y))
    inside = unique_x < max_fpr
    curve_x, curve_y = unique_x[inside], unique_y[inside]
    if len(curve_x) == 0 or curve_x[0] != 0:
        curve_x = np.insert(curve_x, 0, 0.0)
        curve_y = np.insert(curve_y, 0, 0.0)
    curve_x = np.append(curve_x, max_fpr)
    curve_y = np.append(curve_y, endpoint)
    return float(np.clip(trapezoid(curve_y, curve_x) / max_fpr, 0.0, 1.0))


def evaluate_category(labels, scores, masks, maps, threshold):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float64)
    if np.unique(labels).size != 2 or np.unique(masks).size != 2:
        raise ValueError("Test images and masks must each contain both classes.")
    predictions = (scores >= threshold).astype(np.uint8)
    flat_masks = masks.reshape(-1).astype(np.uint8)
    flat_maps = maps.reshape(-1)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "image_precision": float(precision_score(labels, predictions, zero_division=0)),
        "image_recall": float(recall_score(labels, predictions, zero_division=0)),
        "image_f1": float(f1_score(labels, predictions, zero_division=0)),
        "pixel_auroc": float(roc_auc_score(flat_masks, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_masks, flat_maps)),
        "aupro_30pct_fpr": compute_aupro(masks, maps),
    }
