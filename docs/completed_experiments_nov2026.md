# Completed experiment records

## PatchCore (Anomalib 2.7.0)

- Categories: 15/15 completed
- Runtime: 38.99 minutes total; 2.60 minutes/category on average
- Backbone: ImageNet-pretrained Wide ResNet-50-2
- Features: layer2 and layer3
- Coreset sampling ratio: 0.10
- Nearest neighbours: 9
- Train/evaluation batch size: 8/8
- Data-loader workers: 0
- Seed: 42
- Software: Anomalib 2.7.0, PyTorch 2.11.0+cu130

The category CSV contains the recorded image AUROC/F1 and pixel AUROC/F1 values. AP was not present in the retained PatchCore summary file, so it is intentionally not fabricated here.

## MemAE

- Categories: 15/15 completed
- Runtime: 54.97 minutes total; 3.66 minutes/category on average
- Input size: 256×256 RGB
- Memory slots: 256
- Hard-shrinkage threshold: 0.001
- Attention temperature: 0.07
- Maximum epochs: 40; early-stopping patience: 8
- Batch size: 16
- Learning rate: 0.001
- Weight decay: 1e-5
- Validation: held-out 10% of normal training images (normal-only calibration)
- Image threshold percentile: 99th
- Pixel threshold percentile: 99.5th
- Seed: 42

The MemAE category table contains per-category AUROC, pixel AP and F1 values transcribed from the saved experiment summary. Per-category image AP was not available in the displayed retained table and is therefore omitted. Macro averages for all six metrics are recorded separately.

## Caveats

- These runs were produced using separate notebook implementations and should not be treated as an identical controlled comparison in every implementation detail.
- In particular, F1 depends on threshold selection. PatchCore's F1 threshold protocol was not verified to match MemAE's normal-only percentile calibration.
- MemAE's maximum-pixel-score experiment was an exploratory post-hoc test-set ablation. The primary reported result remains the original mean of the largest 1% of reconstruction-error pixels.
- Model weights, raw anomaly maps, per-image predictions and full visualisation folders are retained in Google Drive rather than this repository to avoid committing large artifacts and dataset-derived binary data.
