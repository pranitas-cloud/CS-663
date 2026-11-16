# Repository submission checklist

## Repository and licensing

- [x] Root MIT license added for original project code and documentation, with contributor approval.
- [x] Third-party and MVTec dataset terms documented separately.
- [x] MVTec images, masks, and archives excluded from Git.
- [x] Research papers cited in the references document.
- [ ] All contributors verify that no unlicensed external source code or assets remain.

## Reproducibility

- [x] Python package metadata and core dependency requirements declared.
- [x] Development test requirements declared.
- [x] Anomalib 2.7.0 dependency separated from the base environment.
- [x] Dataset structure validator added.
- [x] CPU test workflow added to GitHub Actions.
- [ ] Run the complete CPU test suite successfully on a clean local clone.
- [ ] Run each full Kaggle notebook from a fresh session and record the resulting workflow URL.
- [ ] Confirm that each Kaggle notebook finds the attached dataset without relying on a personal account path.
- [ ] Confirm that the optional MemAE Drive backup is not required to complete a run.

## Results and reporting

- [x] Completed PatchCore/Anomalib and MemAE result tables committed.
- [x] Macro and per-category tables linked from the README.
- [x] Local PatchCore-style prototype distinguished from the completed Anomalib benchmark.
- [ ] Verify reported metrics against original run artifacts and freeze the final report tables.
- [ ] Record exact Python, PyTorch, Torchvision, CUDA, Anomalib and GPU versions for each completed run.

## Code and notebooks

- [ ] Remove temporary debug output and stale scratch cells from notebooks.
- [ ] Remove obsolete or misleading notebook placeholders, or label them clearly.
- [ ] Confirm all notebook outputs are cleared before final submission.
- [ ] Run syntax/import checks on a clean environment.
- [ ] Check that all documented paths and commands exist in the repository.

## Shared conventions

- Input images: float32, NCHW, RGB, range [0, 1] for RGB pipelines.
- PCA uses grayscale tensors where configured.
- Model fitting uses defect-free training images only.
- Test labels and masks are reserved for evaluation.
- Larger anomaly-map values always mean more anomalous.
