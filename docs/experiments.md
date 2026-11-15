# Experiment plan

1. **Dataset audit:** category counts, image dimensions, defect types, mask coverage, class imbalance.
2. **Sanity checks:** overfit a tiny normal subset; verify output-map alignment; verify good masks are empty.
3. **Baselines:** global PCA, patch PCA, convolutional autoencoder, PatchCore-style detector.
4. **Ablations:** PCA retained variance/components and patch geometry; autoencoder latent width/training duration; PatchCore feature layers, memory-bank budget and neighbour count.
5. **Failure analysis:** relate image/pixel scores and localization quality to defect area fraction, contrast, connected components, bounding-box aspect ratio, perimeter and normalized centroid. Add texture descriptors only after defining and validating them.
6. **Reporting:** per-category tables, macro means, confidence intervals or repeated seeds where feasible, and qualitative true-positive/false-positive/false-negative panels.

Keep a machine-readable CSV/JSON record for each image and run. Never select settings on the official test masks and then report the same test set as unbiased evaluation.
