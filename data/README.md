# Dataset setup

1. Download **MVTec Anomaly Detection (MVTec AD)** from the [official dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2. Extract it under `data/mvtec_anomaly_detection/`.
3. Confirm each category contains `train/good`, `test/<defect_type>`, and `ground_truth/<defect_type>` directories.

The loader expects the original MVTec AD folder names. Images are read as RGB; masks are grayscale and binarized. Good test images have no mask and receive an all-zero ground-truth mask.

**Do not upload raw images or masks to GitHub.** The repository .gitignore excludes contents of this directory except this README.
