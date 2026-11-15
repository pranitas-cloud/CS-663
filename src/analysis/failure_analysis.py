"""Join per-image predictions with defect descriptors for failure analysis."""
import pandas as pd


def build_failure_table(records):
    """records: iterable of dicts containing metrics, properties and identifiers."""
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return frame
    if "image_auroc" in frame and "pixel_auroc" in frame:
        frame["image_detected"] = frame["image_auroc"] >= 0.5
        frame["localization_adequate"] = frame["pixel_auroc"] >= 0.5
    return frame


def grouped_summary(frame, by="defect_type"):
    if frame.empty or by not in frame:
        return pd.DataFrame()
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return frame.groupby(by, dropna=False)[numeric].mean().reset_index()
