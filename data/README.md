# Dataset setup

Download **MVTec AD (not MVTec AD 2)** from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The dataset has 15 object/texture categories with defect-free training images, test images, and pixel-level anomaly masks.

The dataset is licensed under **CC BY-NC-SA 4.0** and is for non-commercial use. Check the license before sharing or reusing it.

## Expected directory layout

Extract the archive so the root directly contains category folders:

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
└── ... (15 categories)
```

In Colab, store the extracted folder in Google Drive and set `DATA_ROOT` in `notebooks/01_dataset_exploration.ipynb` to the folder containing `bottle/`, `cable/`, etc.

## Split protocol

For each category, the loader makes a reproducible 90/10 split of the official `train/good` images (seed 42):
- `split="train"`: normal training images.
- `split="val"`: held-out normal validation images.
- `split="test"`: official test images, normal and anomalous, with masks.

Only defect-free training images are used for fitting and validation. Test images are kept separate. Images are RGB float tensors in [0, 1], resized to 256 × 256 by default. Masks use nearest-neighbour resizing and are binarized.

**Do not upload raw images or masks to GitHub.** Dataset contents are excluded by .gitignore.
