"""Image- and pixel-level metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels).ravel(), np.asarray(scores).ravel()
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def evaluate_batch(image_labels, image_scores, masks, anomaly_maps):
    image = binary_metrics(image_labels, image_scores)
    pixel = binary_metrics(np.asarray(masks).ravel(), np.asarray(anomaly_maps).ravel())
    return {"image_auroc": image["auroc"], "image_ap": image["average_precision"],
            "pixel_auroc": pixel["auroc"], "pixel_ap": pixel["average_precision"]}
