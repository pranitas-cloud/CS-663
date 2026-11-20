# Methodology

## Objective

Evaluate Global Principal Component Analysis (PCA) as an unsupervised reconstruction-based anomaly detector on the 15-category MVTec AD dataset.

## Preprocessing and data split

- Resize images to 256 × 256 and convert RGB images to grayscale in [0, 1].
- Flatten each image into a 65,536-dimensional feature vector.
- Split only normal train/good images deterministically into 90% training and 10% validation.
- Fit PCA using the training split only. Use validation images only to calibrate the image-level threshold.
- Evaluate on the official test split and its ground-truth masks.
- Resize binary masks with max pooling to preserve small defect regions.

## Global PCA

Let x be a flattened image, μ the mean training image, and Vₖ the first k principal directions. The reconstruction is

$$
z = V_k^T(x-\mu), \qquad \hat{x} = \mu + V_k z.
$$

The pixel anomaly map is the squared reconstruction residual, reshaped to image dimensions:

$$
A(i,j) = (x(i,j)-\hat{x}(i,j))^2.
$$

The implementation computes the PCA basis using economy SVD of the centred training matrix. The experiment retains enough components to explain 90%, 95%, or 98% of training variance.

## Experimental factors

| Factor | Values |
|---|---|
| Retained variance | 90%, 95%, 98% |
| Gaussian smoothing sigma | 0, 0.75 |
| Image-level score | Maximum residual, mean of top 1% residuals |
| Categories | All 15 MVTec AD categories |

This gives 12 configurations per category and 180 category/configuration evaluations.

## Evaluation

- Image AUROC and image average precision evaluate image-level ranking.
- Image precision, recall, and F1 use a threshold set to the 95th percentile of normal validation image scores.
- Pixel AUROC and pixel average precision evaluate pixel-level anomaly ranking.
- AUPRO is integrated up to 30% pixel false-positive rate using connected components in the ground-truth masks.

## Limitations

Global PCA uses one position-dependent subspace per category. Flattening does not model local texture or semantic structure explicitly. Small defects may have little influence on the global representation, while normal texture variation or misalignment can produce large residuals. Reconstruction error is an anomaly signal, not a defect-specific classifier.