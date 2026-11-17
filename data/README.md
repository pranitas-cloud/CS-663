# Dataset

This project uses MVTec AD (15 categories). Download the dataset from the official [MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and review its CC BY-NC-SA 4.0 terms before use or redistribution.

Place the extracted dataset so the 15 category folders are directly inside:

`data/mvtec_anomaly_detection/`

Each category must contain `train/good`, `test`, and `ground_truth`. The dataset is not included in this repository.

The code creates a deterministic validation split from `train/good` only. The official test split is reserved for final evaluation.
