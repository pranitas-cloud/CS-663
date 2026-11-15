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
