# Step 1 — Project skeleton

- [x] Shared detector interface documented (`fit(train_images)`, `predict(images) -> N×H×W`)
- [x] Package structure for data, methods, analysis, evaluation, notebooks, configs, docs, results
- [x] Dependencies listed
- [x] Dataset and generated artifacts excluded from git
- [ ] Create local environment and install dependencies
- [ ] Run import/syntax smoke checks locally
- [ ] Team agrees on image tensor convention and anomaly-map shape

## Shared conventions

- Input images: float32, NCHW, RGB, range [0, 1].
- `fit`: only normal/defect-free training images.
- `predict`: return `numpy.ndarray` with shape (N, H, W).
- Larger anomaly-map values always mean more anomalous.
- Test labels and masks are reserved for evaluation.
