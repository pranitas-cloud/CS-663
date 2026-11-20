# Presentation Brief

## Project

**Global PCA for Industrial Visual Anomaly Detection**  
CS 663 — Digital Image Processing  
Dataset: MVTec AD, 15 categories

## Objective

Learn a low-dimensional representation of defect-free images and detect anomalies through squared reconstruction error, without fitting on anomalous training examples.

## Method

1. Resize each image to 256 × 256 and convert to grayscale.
2. Flatten each image into a 65,536-dimensional vector.
3. Fit one PCA model per category using only normal train/good images.
4. Retain components explaining 90%, 95%, or 98% of training variance.
5. Project and reconstruct test images.
6. Use squared reconstruction residuals as pixel anomaly maps.
7. Compare raw and Gaussian-smoothed maps (sigma 0.75).
8. Score images using maximum residual or mean of the top 1% residual pixels.

The executed notebook uses economy SVD on the centred training matrix.

## Data and evaluation protocol

- Deterministic 90/10 split of normal train/good images into PCA training and normal validation.
- Validation scores set the image threshold at the 95th percentile.
- Official test images and masks are used for evaluation.
- All 15 MVTec AD categories are included.
- 3 retained-variance settings × 2 smoothing settings × 2 image-score settings = 12 configurations per category, 180 evaluations total.

## Main macro results

Best image AUROC configuration: 98% variance, sigma 0.75, top-1% score.

| Metric | Macro result |
|---|---:|
| Image AUROC | 0.8005 |
| Image AP | 0.9079 |
| Image precision | 0.9163 |
| Image recall | 0.4401 |
| Image F1 | 0.5315 |
| Pixel AUROC | 0.8314 |
| Pixel AP | 0.2663 |
| AUPRO@30% FPR | 0.6466 |

The pixel metrics are based on anomaly maps and do not depend on the image-level scoring rule.

## Ablation observations

- Gaussian smoothing improves pixel AUROC from 0.7748 (98%, sigma 0) to 0.8314 (98%, sigma 0.75).
- At 98% variance and sigma 0.75, top-1% scoring gives image AUROC 0.8005 versus 0.7963 for max scoring.
- The validation-derived threshold gives high precision (0.9163) but low recall (0.4401), showing a conservative image-level operating point.

## Limitations

- A single global subspace is position-dependent and sensitive to alignment.
- Flattening does not explicitly encode local texture or semantic structure.
- Small defects may contribute little to global variance.
- Normal texture variation can create large reconstruction residuals.
- Reconstruction error is not a defect-specific classifier.

## Suggested slide sequence

1. Problem statement and MVTec AD dataset
2. Global PCA intuition and reconstruction equations
3. Data split and preprocessing
4. Anomaly-map generation
5. Experimental design and ablations
6. Quantitative macro results
7. Example: input, raw residual, smoothed residual, ground-truth mask
8. Category variation, limitations, and possible improvements

## Accuracy notes

- The saved notebook's numerical outputs are preserved; its stale setup-cell output was removed.
- The executed experiment uses SVD. Do not describe its reported metrics as Gram-matrix results.
- Full category-level and macro tables are displayed in the notebook outputs. CSV files are not committed.
- Do not claim comparisons against PatchCore, autoencoders, or other methods; they are not part of this repository.
- Do not describe the reported metrics as state of the art.