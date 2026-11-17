from pathlib import Path

import numpy as np
from PIL import Image

from scripts.validate_dataset import validate_category


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_validate_category_checks_structure_and_masks(tmp_path):
    category_root = tmp_path / "bottle"
    save_png(
        category_root / "train" / "good" / "000.png",
        np.zeros((32, 32, 3), dtype=np.uint8),
    )
    save_png(
        category_root / "test" / "good" / "000.png",
        np.zeros((32, 32, 3), dtype=np.uint8),
    )
    save_png(
        category_root / "test" / "crack" / "001.png",
        np.ones((32, 32, 3), dtype=np.uint8),
    )
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    save_png(
        category_root / "ground_truth" / "crack" / "001_mask.png",
        mask,
    )

    result = validate_category(tmp_path, "bottle")
    assert result == {
        "train_good": 1,
        "test_total": 2,
        "test_good": 1,
        "test_defective": 1,
        "masks_checked": 1,
    }
