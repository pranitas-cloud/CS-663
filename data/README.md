# Dataset setup

This project uses **MVTec AD** (not MVTec AD 2), an industrial anomaly detection benchmark with 15 object and texture categories, defect-free training images, test images, and pixel-level anomaly masks.

Download it from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The dataset is released under **CC BY-NC-SA 4.0**. Attribution is required, commercial use is not permitted, and adaptations are subject to the ShareAlike condition. Read the complete terms before using or sharing the dataset. The dataset is not included in this repository.

## Expected directory layout

Extract the archive so the root directly contains the category folders:

```text
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/*.png
│   ├── test/
│   │   ├── good/*.png
│   │   └── <defect_type>/*.png
│   └── ground_truth/<defect_type>/*_mask.png
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
```

For local runs, the recommended location is:

```text
data/mvtec_anomaly_detection/
```

You may store it elsewhere. Pass the directory containing the 15 category folders to the dataset validator or configure the data root in the relevant notebook/configuration.

## Validate the download

From the repository root:

```bash
python scripts/validate_dataset.py --data-root data/mvtec_anomaly_detection
```

To check one category:

```bash
python scripts/validate_dataset.py --data-root data/mvtec_anomaly_detection --category bottle
```

The validator checks the category layout, PNG readability, square image geometry, and matching ground-truth masks for defective test images. It does not modify the dataset.

## Split protocol

For the repository dataset loader, each category's official `train/good` images are deterministically split into training and validation subsets using seed 42 and a configurable validation fraction (default 10%):
- `split="train"`: normal training images.
- `split="val"`: held-out normal validation images.
- `split="test"`: official test images, normal and anomalous, with masks.

Only defect-free training images are used for fitting and validation. Official test images are held out for final evaluation. The preprocessing pipeline converts images to tensors and resizes them; masks use nearest-neighbour resizing by default and are binarized.

## Kaggle notebook runs

For the full Anomalib PatchCore and MemAE notebooks, attach your separately obtained MVTec dataset as a Kaggle input. The notebooks search the Kaggle input mount for a dataset folder or archive. If the dataset is mounted with a nonstandard name or nested layout, adjust the dataset-discovery cell and confirm the printed root before training.

## Data handling rules

- Do not upload raw MVTec images, masks, or dataset archives to GitHub.
- Do not add generated figures containing MVTec images to the repository without reviewing whether redistribution is allowed by the dataset license.
- Do not assume that the repository's MIT license applies to MVTec AD.
- Keep local data under the ignored `data/` directory or use an external data mount.
