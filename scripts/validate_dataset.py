from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
)


def png_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.png") if path.is_file())


def inspect_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    if width != height:
        raise ValueError(f"Expected a square MVTec image or mask: {path} ({width}x{height})")
    return width, height


def validate_category(root: Path, category: str) -> dict[str, int]:
    base = root / category
    train_folder = base / "train" / "good"
    test_folder = base / "test"
    ground_truth = base / "ground_truth"

    if not train_folder.is_dir():
        raise FileNotFoundError(f"Missing normal training folder: {train_folder}")
    if not test_folder.is_dir():
        raise FileNotFoundError(f"Missing test folder: {test_folder}")

    train_images = png_files(train_folder)
    if not train_images:
        raise ValueError(f"No PNG images found in {train_folder}")

    test_images = 0
    normal_images = 0
    defective_images = 0
    masks_checked = 0

    defect_folders = sorted(path for path in test_folder.iterdir() if path.is_dir())
    if not defect_folders:
        raise ValueError(f"No test subfolders found in {test_folder}")

    for defect_folder in defect_folders:
        images = png_files(defect_folder)
        if not images:
            raise ValueError(f"No PNG images found in {defect_folder}")

        for image_path in images:
            inspect_image(image_path)
            test_images += 1

            if defect_folder.name == "good":
                normal_images += 1
                continue

            defective_images += 1
            mask_path = ground_truth / defect_folder.name / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing ground-truth mask for {image_path}: {mask_path}")
            image_size = inspect_image(image_path)
            mask_size = inspect_image(mask_path)
            if image_size != mask_size:
                raise ValueError(f"Image/mask size mismatch: {image_path} and {mask_path}")
            masks_checked += 1

    for image_path in train_images:
        inspect_image(image_path)

    return {
        "train_good": len(train_images),
        "test_total": test_images,
        "test_good": normal_images,
        "test_defective": defective_images,
        "masks_checked": masks_checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the expected MVTec AD directory structure and image-mask pairs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/mvtec_anomaly_detection"),
        help="Directory that directly contains the 15 MVTec category folders.",
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        help="Validate only one category. By default, all 15 categories are required.",
    )
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")

    categories = [args.category] if args.category else list(CATEGORIES)
    available = {path.name for path in root.iterdir() if path.is_dir()}
    missing = [category for category in categories if category not in available]
    if missing:
        parser.error("Missing category folders: " + ", ".join(missing))

    print(f"Dataset root: {root}")
    print(f"{'category':<14} {'train':>7} {'test':>7} {'good':>7} {'defect':>7} {'masks':>7}")
    totals = {key: 0 for key in ("train_good", "test_total", "test_good", "test_defective", "masks_checked")}

    for category in categories:
        counts = validate_category(root, category)
        for key in totals:
            totals[key] += counts[key]
        print(
            f"{category:<14} {counts['train_good']:>7} {counts['test_total']:>7} "
            f"{counts['test_good']:>7} {counts['test_defective']:>7} {counts['masks_checked']:>7}"
        )

    if not args.category and len(available.intersection(CATEGORIES)) != len(CATEGORIES):
        parser.error("Expected all 15 MVTec AD categories.")

    print(
        f"{'TOTAL':<14} {totals['train_good']:>7} {totals['test_total']:>7} "
        f"{totals['test_good']:>7} {totals['test_defective']:>7} {totals['masks_checked']:>7}"
    )
    print("Dataset structure and image-mask pairs passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
