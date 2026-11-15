"""Save anomaly-map overlays for qualitative inspection."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_anomaly_panel(image, mask, anomaly_map, output_path, title=""):
    image = np.asarray(image).transpose(1, 2, 0) if np.asarray(image).ndim == 3 and np.asarray(image).shape[0] in (1, 3) else np.asarray(image)
    mask, anomaly_map = np.squeeze(mask), np.squeeze(anomaly_map)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(image, 0, 1)); axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground truth")
    axes[2].imshow(anomaly_map, cmap="inferno"); axes[2].set_title("Anomaly score")
    for ax in axes: ax.axis("off")
    fig.suptitle(title); fig.tight_layout()
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)
