# Global PCA for Industrial Visual Anomaly Detection

**CS 663 — Digital Image Processing | MVTec AD**

A global principal component analysis baseline for unsupervised anomaly detection on the 15-category MVTec AD benchmark.

## Method

Each grayscale image is flattened into one feature vector. PCA is fitted on defect-free training images only. Test images are projected into the retained subspace and reconstructed; squared reconstruction residuals form pixel-level anomaly maps.

The executed experiment compares 90%, 95% and 98% retained variance, Gaussian smoothing (sigma 0 and 0.75), and max versus top-1% image scoring.

## Full-run results

The committed notebook contains the executed 15-category run (180 category/configuration rows). Best macro image AUROC was **0.8005** (98% variance, sigma 0.75, top-1% score). Best macro pixel AUROC was **0.8314** (98% variance, sigma 0.75); corresponding pixel AP was **0.2663** and AUPRO@30% FPR was **0.6466**. These results describe this experiment and are not state-of-the-art claims.

## Repository layout

- `notebooks/global_pca.ipynb`: executed experiment, metrics and visualization.
- `src/pca/`: PCA and linear algebra implementation.
- `src/data/`: MVTec loading and preprocessing.
- `src/evaluation.py`: evaluation metrics.
- `docs/`: methodology and references.
- `results/`: output location for generated tables and plots.

## Setup

Install dependencies with `pip install -r requirements.txt`. Download MVTec AD and place its 15 category folders under `data/mvtec_anomaly_detection/`. The dataset is not included. In Colab, the notebook uses the configured Google Drive dataset path; set `MVTEC_DATA_ROOT` if needed.

## Evaluation protocol

A deterministic validation split is taken from `train/good` only. The image threshold is the 95th percentile of normal validation scores. Official test images and masks are reserved for final metrics.

## License

The code is distributed under this repository's license. MVTec AD has separate CC BY-NC-SA 4.0 terms. Do not upload dataset images or masks.
