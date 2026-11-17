# Methodology

## Task formulation
Train on defect-free images only. At test time produce (1) an image anomaly score and (2) a dense pixel anomaly map. Ground-truth masks are used only for evaluation and post-hoc defect characterization.

## Methods
### Global PCA
Resize images to a fixed resolution, convert to grayscale and flatten each image into one feature vector. Subtract the mean calculated from normal training images. Since the pixel-space covariance matrix is too large at the working resolution, use the equivalent sample-space matrix X X^T and obtain its eigenvectors with NumPy's symmetric eigensolver. Map the selected eigenvectors back to pixel space to obtain the principal components. Reconstruct images from the retained components and use squared reconstruction residual as the pixel anomaly score. The implementation is written directly in the notebook and does not use a prebuilt PCA estimator.

### Patch PCA
Extract overlapping patches from normal training images. Sample a reproducible subset of patches, subtract the training-patch mean, calculate the feature covariance matrix and obtain its eigenvectors with NumPy's symmetric eigensolver. Reconstruct each patch from the retained components. Project each patch's squared reconstruction residual back to its image location and average overlapping error contributions. Patch size, stride, retained variance and sample count are experimental factors. The implementation is written directly in the notebook and does not use a prebuilt PCA estimator.

### Convolutional autoencoder
Train a compact convolutional encoder-decoder from scratch on normal images using pixelwise MSE. At inference, aggregate channelwise squared reconstruction residuals into an anomaly map. This baseline may reconstruct some defects well; document that limitation.

### PatchCore-style method
Use ImageNet-pretrained Wide-ResNet-50-2 intermediate feature maps. Concatenate aligned layer-2/layer-3 features, flatten spatial positions into patch descriptors, store a representative normal memory bank, and score test descriptors by nearest-neighbour cosine distance. Upsample patch scores to image resolution. The current memory-bank sampler is random; implement and compare a greedy coreset selection for a closer PatchCore reproduction.

## Evaluation protocol
- Train on `train/good` only.
- Evaluate on the official test split and ground-truth masks.
- Report per-category image AUROC, pixel AUROC, and average precision; macro-average across categories.
- Preserve original masks with nearest-neighbour resizing. Do not use test masks to select hyperparameters.
- State image resolution, random seed, preprocessing, backbone weights, memory-bank size and all model settings with every result.
