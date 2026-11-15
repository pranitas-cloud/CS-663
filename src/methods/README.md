# Detector interface

All detector implementations must expose:

- `fit(train_images)`: receives only defect-free training images.
- `predict(images) -> anomaly_maps`: returns a NumPy array of shape `(N, H, W)`; larger values mean more anomalous.

Input convention: float32 NCHW RGB images scaled to [0, 1]. Ground-truth masks and test labels must never be passed to `fit`.

The PCA, CAE and PatchCore modules in the existing folders will be adapted to this interface in their implementation steps. This file establishes the contract before parallel development.
