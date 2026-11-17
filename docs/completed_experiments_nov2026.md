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

The committed MemAE category CSV contains per-category AUROC and pixel AP values but does not contain an `image_AP` column, although the macro CSV records `image_AP = 0.841480`. The original F1 header also declared image and pixel F1 separately, but each row contained only one F1 value; the category table now labels it `f1_score_field_unverified` rather than guessing its meaning. The macro CSV records image F1 0.425354 and pixel F1 0.125190, but those two values cannot currently be reproduced from the category CSV. The original per-category JSON metrics are required to reconcile these fields.

## Caveats

- These runs were produced using separate notebook implementations and should not be treated as an identical controlled comparison in every implementation detail.
- In particular, F1 depends on threshold selection. PatchCore's F1 threshold protocol was not verified to match MemAE's normal-only percentile calibration.
- MemAE's maximum-pixel-score experiment was an exploratory post-hoc test-set ablation. The primary reported result remains the original mean of the largest 1% of reconstruction-error pixels.
- Model weights, raw anomaly maps, per-image predictions and full visualisation folders are retained in Google Drive rather than this repository to avoid committing large artifacts and dataset-derived binary data.
