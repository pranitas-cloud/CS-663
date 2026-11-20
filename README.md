# Global PCA for Industrial Visual Anomaly Detection

**CS 663 — Digital Image Processing | MVTec AD**

An unsupervised, reconstruction-based anomaly-detection baseline using one global PCA subspace per MVTec AD category.

## Method

Normal training images are resized to 256 × 256, converted to grayscale, flattened, and used to fit PCA. Test images are projected into the retained subspace and reconstructed. Squared reconstruction residuals form pixel-level anomaly maps.

## Experiment

The executed notebook evaluates all 15 MVTec AD categories across 12 configurations: retained variance (90%, 95%, 98%), Gaussian smoothing (sigma 0, 0.75), and image scoring (maximum, top-1% mean). This produces 180 category/configuration evaluations.

## Results

Best macro image AUROC: **0.8005** (98% variance, sigma 0.75, top-1% score). Best macro pixel AUROC: **0.8314** (98% variance, sigma 0.75); pixel AP: **0.2663**; AUPRO@30% FPR: **0.6466**. The notebook contains the full category-level and macro tables, along with the saved visualization. These are results of this experiment, not state-of-the-art claims.

## Repository

- notebooks/global_pca.ipynb — executed experiment, saved outputs, result tables, and visualization.
- src/pca/ — PCA model and linear algebra routines.
- src/data/ — MVTec loading and preprocessing.
- src/evaluation.py — image- and pixel-level metrics.
- configs/default.yaml — experiment defaults.
- docs/methodology.md — mathematical formulation and evaluation protocol.
- docs/presentation_brief.md — concise presentation outline and verified results.
- docs/references.md — references.

## Reproduction

Install dependencies with pip install -r requirements.txt. Download MVTec AD separately and place the 15 category folders under data/mvtec_anomaly_detection/. The dataset is not included in this repository. In Colab, the notebook defaults to MyDrive/CS663_Shared/datasets/mvtec_anomaly_detection; set MVTEC_DATA_ROOT to use a different location.

The deterministic validation split uses only train/good. The official test split is reserved for final evaluation.

## Scope and limitations

This repository contains the Global PCA experiment only. It does not include the MVTec dataset or implementations of other anomaly-detection methods. Global PCA is position-sensitive and does not explicitly model local texture or semantic features.

## License

Code is distributed under this repository's license. MVTec AD is separately licensed under CC BY-NC-SA 4.0. Do not commit dataset images or masks.