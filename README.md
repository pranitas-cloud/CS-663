# Global PCA for Industrial Visual Anomaly Detection

A standalone implementation of Global Principal Component Analysis for unsupervised anomaly detection on the MVTec AD benchmark.

## Method

The model learns a low-dimensional appearance subspace from defect-free training images. Test images are projected into this subspace and reconstructed. Squared reconstruction residuals form a pixel-level anomaly map. The image-level score is the average of the highest-scoring one percent of pixels.

The experiment compares three retained-variance targets (90%, 95%, 98%), raw and Gaussian-smoothed residual maps, and maximum versus top-one-percent image scoring. The PCA objective remains unchanged across these ablations.

## Repository layout

- `notebooks/global_pca.ipynb`: complete experiment and evaluation.
- `src/pca/`: Global PCA implementation and linear algebra.
- `src/data/`: MVTec loading and preprocessing.
- `src/evaluation.py`: image, pixel and region-based metrics.
- `docs/`: methodology and references.
- `results/`: generated tables and visualizations.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download MVTec AD and place the extracted category folders under `data/mvtec_anomaly_detection/`. See [data/README.md](data/README.md).

In Google Colab, mount Google Drive and set `DATA_ROOT` in the first notebook code cell to the directory containing the 15 category folders.

## Run

Open `notebooks/global_pca.ipynb` and run all cells. The default setting evaluates all 15 categories. Set `RUN_ALL_CATEGORIES = False` for a single-category smoke test.

The notebook writes category-level and macro-average CSV files to `results/tables/` and saves a representative visualization to `results/plots/`.

## Evaluation protocol

PCA is fitted on `train/good` only. A deterministic 90/10 split of those normal images provides the validation scores used to set the image threshold at the 95th percentile. The official test split and masks are used only for final evaluation.

No benchmark numbers are claimed in this repository until the complete notebook has been run and its generated result tables have been reviewed.

## License and dataset terms

The source code is distributed under the repository license. MVTec AD is a separately licensed dataset; its CC BY-NC-SA 4.0 terms apply to the data, not the code. Do not upload raw dataset images or masks to GitHub.
