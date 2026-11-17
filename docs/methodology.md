# Methodology

## Global PCA

Each preprocessed image is represented as one flattened grayscale vector. The training matrix contains only defect-free images from `train/good`. The training mean is subtracted before computing principal directions.

The implementation uses an economy singular value decomposition of the centered sample matrix. This obtains the same principal directions as eigendecomposition of the feature covariance matrix while avoiding construction of a large pixel-by-pixel covariance matrix.

For a retained basis (V_k), an image (x) is projected and reconstructed as:

[
z = V_k^T(x-\mu), \qquad \hat{x}=\mu+V_kz.
]

The pixel anomaly map is the squared reconstruction residual ((x-\hat{x})^2). The image score is the mean of the largest one percent of residual values. A max-score alternative is included in the ablation.

## Data protocol

Images are resized to 256 by 256 with anti-aliased bilinear interpolation and converted to grayscale in the range [0,1]. Masks are binarized and downsampled with max pooling to preserve small defect regions.

A deterministic 90/10 split of `train/good` is used for fitting and normal-only threshold calibration. The image threshold is the 95th percentile of normal validation scores. The official test split is used only for final metrics.

## Ablations

The experiment compares retained variance targets of 90%, 95%, and 98%; no smoothing versus Gaussian smoothing with sigma 0.75; and maximum versus top-one-percent image scoring. These settings are fixed before test evaluation. The results are reported as controlled comparisons, not as guaranteed improvements.

## Metrics

Image AUROC and average precision evaluate image ranking. Pixel AUROC and average precision evaluate pixel ranking. Image precision, recall and F1 use the validation-derived threshold. AUPRO is approximated on a quantile-spaced threshold grid up to 30% pixel false-positive rate.

Global PCA models whole-image variation and is sensitive to object alignment and spatial position. Localized texture changes may not be well represented by a global linear subspace.
