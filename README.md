# Industrial Anomaly Detection

**CS 663 — Digital Image Processing | Project**

A reproducible study of unsupervised industrial visual anomaly detection on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). The project compares three complementary approaches:

1. **PCA reconstruction** — fit a low-dimensional normality model using only defect-free training images.
2. **Convolutional autoencoder** — train an encoder-decoder from scratch to reconstruct normal images.
3. **PatchCore-style detector** — extract pretrained CNN patch features, build a normal-feature memory bank, and score test patches using nearest neighbours.

The project evaluates image-level detection and pixel-level localization, then studies which defect characteristics are associated with successes and failures.

## Research questions

- How do reconstruction-based and feature-memory approaches compare on image-level detection and pixel-level localization?
- How do PCA dimension, autoencoder capacity, feature layers, memory-bank size and nearest-neighbour settings affect results?
- How do defect size, contrast, texture, shape and location relate to detection/localization performance?
- Which failure modes are shared across methods, and which are method-specific?

## Repository layout

```text
configs/       Experiment configuration
data/          Dataset instructions (raw data is not committed)
src/data/      MVTec loading and preprocessing
src/pca/       Global and patch PCA baselines
src/autoencoder/ CNN autoencoder and training loop
src/patchcore/ Feature extraction, memory bank and detector
src/evaluation/ Metrics and visualizations
src/analysis/  Defect-property and failure analysis
notebooks/     Ordered experiment notebooks
experiments/   Run outputs/checkpoints (ignored by git)
results/       Generated maps, plots and tables (ignored by git)
docs/          Methodology, experiment protocol and references
```

## Setup

Python 3.10+ is recommended.

```bash
git clone https://github.com/pranitas-cloud/CS-663.git
cd CS-663
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

For a CUDA-enabled PyTorch installation, install the build matching your CUDA version from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Dataset

Download and extract MVTec AD yourself; see [data/README.md](data/README.md). Expected layout:

```text
data/mvtec_anomaly_detection/
  bottle/
    train/good/
    test/good/
    test/broken_large/
    ground_truth/broken_large/
  ...
```

Do not commit the dataset. Set `data.root` in `configs/default.yaml` or pass `--data-root` to scripts.

## First checks

```bash
python -m src.data.dataset --data-root data/mvtec_anomaly_detection --category bottle
python -m src.pca.global_pca --data-root data/mvtec_anomaly_detection --category bottle
python -m src.autoencoder.train --data-root data/mvtec_anomaly_detection --category bottle --epochs 2
python -m src.patchcore.detector --data-root data/mvtec_anomaly_detection --category bottle
```

These commands are minimal smoke-test entry points, not final benchmark runs. Tune settings in the config and record every run's parameters.

## Methodological notes

- Training uses only the MVTec `train/good` split. Test images and masks are reserved for evaluation.
- Metrics are computed per category and then macro-averaged; report both image AUROC and pixel AUROC. Add AUPRO if the final protocol supports it.
- The PCA implementation is a transparent baseline over resized grayscale images. Global PCA may miss small/localized defects; patch PCA is included to test local modelling.
- The autoencoder is trained from scratch. Reconstruction error is a baseline anomaly score, not a calibrated probability.
- PatchCore uses ImageNet-pretrained Wide ResNet features by default. The feature extractor is pretrained; patch aggregation, memory bank and scoring are implemented in this repository.
- Avoid tuning on test masks. Use training-only decisions or a clearly separated validation protocol.

See [docs/methodology.md](docs/methodology.md), [docs/experiments.md](docs/experiments.md), and [docs/references.md](docs/references.md).


## Completed experiments (November 2026)

### PatchCore — Anomalib run

The completed 15-category run used Anomalib 2.7.0 with PyTorch 2.11.0+cu130, an ImageNet-pretrained `wide_resnet50_2` backbone, feature layers `layer2` and `layer3`, coreset sampling ratio 0.10, and 9 nearest neighbours. Batch sizes were 8 for training and evaluation, with 0 data-loader workers and random seed 42. The run took 38.99 minutes in total.

Macro metrics are recorded in [results/tables/patchcore_anomalib_macro.csv](results/tables/patchcore_anomalib_macro.csv), with category-level values in [results/tables/patchcore_anomalib_categories.csv](results/tables/patchcore_anomalib_categories.csv).

### MemAE — memory-augmented reconstruction run

A separate MemAE implementation was trained independently for all 15 MVTec AD categories, using only normal training images and a held-out normal validation subset for threshold calibration. Inputs were resized to 256×256. The run used a memory size of 256, hard-shrinkage threshold 1e-3, temperature 0.07, maximum 40 epochs, early-stopping patience 8, batch size 16, learning rate 1e-3, and seed 42. Total measured category runtime was approximately 54.97 minutes.

Macro metrics are recorded in [results/tables/memae_macro.csv](results/tables/memae_macro.csv), and category-level AUROC/F1/runtime values are in [results/tables/memae_categories.csv](results/tables/memae_categories.csv).

### Interpretation and comparability

- PatchCore's recorded macro image AUROC and pixel AUROC were 0.9823 and 0.9801. Its recorded image F1 and pixel F1 were 0.9715 and 0.5801.
- MemAE's recorded macro image AUROC, image AP, image F1, pixel AUROC, pixel AP and pixel F1 were 0.6722, 0.8415, 0.4254, 0.7381, 0.1448 and 0.1252.
- F1 thresholds and evaluation implementations were not established to be identical across these two runs. Treat AUROC/AP as the cleaner ranking comparisons; interpret F1 cautiously.
- A post-hoc MemAE image-score ablation found maximum pixel error gave macro image AUROC 0.6967 and image AP 0.8535, compared with 0.6722 and 0.8415 for the original top-1%-mean score. This was exploratory and selected after inspecting test performance; it is not the primary reported MemAE result.
- Model checkpoints, prediction maps and raw category artifacts are stored in the associated Google Drive experiment folders, not in Git. This repository contains concise result tables and experiment metadata only.
