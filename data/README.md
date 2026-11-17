# Dataset

This project uses the 15-category MVTec AD dataset. Download it from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and review its CC BY-NC-SA 4.0 terms.

Place the extracted category folders under `data/mvtec_anomaly_detection/`. Each category must contain `train/good`, `test`, and `ground_truth`. The dataset is not included in this repository.

The experiment creates a deterministic validation split from `train/good` only. The official test split is reserved for final evaluation.
