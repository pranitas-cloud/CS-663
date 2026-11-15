# Methodology

## Task formulation
Train on defect-free images only. At test time produce (1) an image anomaly score and (2) a dense pixel anomaly map. Ground-truth masks are used only for evaluation and post-hoc defect characterization.

## Methods
### Global PCA
Resize images to a fixed resolution, convert to grayscale, flatten, and fit a low-rank subspace using SVD. Reconstruct test images and use squared reconstruction residual as a pixel score. This is a deliberately simple baseline.

### Patch PCA
Extract overlapping patches from normal training images, fit a shared PCA subspace over flattened patches, and project patch reconstruction errors back to an image-sized score map. Patch size and stride are experimental factors.

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
