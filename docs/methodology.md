# Methodology

## Global PCA

Each preprocessed image is represented as one flattened grayscale vector. The training mean is subtracted before computing principal directions using economy SVD of the centered sample matrix.

For a retained basis (V_k), an image (x) is projected and reconstructed:

[
z=V_k^T(x-mu), qquad hat{x}=mu+V_kz.
]

The pixel anomaly map is the squared reconstruction residual, ((x-hat{x})^2).

## Data protocol

Images are resized to 256 × 256 and converted to grayscale in [0,1]. PCA is fitted using only normal `train/good` images. A deterministic 90/10 split of those images is used for fitting and normal-only threshold calibration. The image threshold is the 95th percentile of validation scores. Official test images and masks are used only for final evaluation.

Masks are binarized and downsampled using max pooling to preserve small defect regions.

## Ablations

- Retained variance: 90%, 95%, 98%.
- Residual smoothing: sigma 0 and 0.75.
- Image scoring: maximum and mean of the top 1% residual pixels.

## Metrics

Image AUROC and average precision measure image ranking. Pixel AUROC and average precision measure pixel ranking. Image precision, recall and F1 use the validation-derived threshold. AUPRO is evaluated up to 30% pixel false-positive rate using a quantile-spaced threshold grid.

Global PCA is position-sensitive and may struggle with local texture changes and object misalignment.
