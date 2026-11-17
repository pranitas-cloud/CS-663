# Repository submission checklist

## Repository and licensing

- [x] Root MIT license added for original project code and documentation, with contributor approval.
- [x] Third-party and MVTec dataset terms documented separately.
- [x] MVTec images, masks, and archives excluded from Git.
- [x] Research papers cited in the references document.
- [ ] All contributors verify that no unlicensed external source code, pretrained assets, or figures remain.

## Reproducibility

- [x] Python package metadata and core dependency requirements declared.
- [x] Development test requirements declared.
- [x] Anomalib 2.7.0 dependency separated from the base environment.
- [x] Dataset structure validator added.
- [x] CPU test workflow added to GitHub Actions.
- [x] Dataset root made configurable instead of relying on a contributor-specific Drive path.
- [ ] Run the complete CPU test suite successfully on a clean local clone and inspect the GitHub Actions result.
- [ ] Run each full Kaggle notebook from a fresh session and record the resulting workflow URL.
- [ ] Confirm that the optional MemAE Drive backup is not required to complete a run.
- [ ] Generate and commit a fully resolved environment lock file after validating the clean installation on the target runtime.

## Results and reporting

- [x] Completed PatchCore/Anomalib and MemAE result tables committed.
- [x] Macro and per-category tables linked from the README.
- [x] Local PatchCore-style prototype distinguished from the completed Anomalib benchmark.
- [x] PatchCore macro metrics in the record notebook reconciled with the committed macro table.
- [ ] Verify reported metrics against original run artifacts and freeze the final report tables.
- [ ] Record exact Python, PyTorch, Torchvision, CUDA, Anomalib and GPU versions for each completed run.

## Code and notebooks

- [x] Notebook outputs cleared and execution counters reset.
- [x] Code-cell comments and stale saved output removed; explanatory documentation retained in Markdown/docstrings.
- [x] Old account-specific Drive and Kaggle dataset paths removed from notebook code.
- [x] Former notebook placeholders labelled as smoke checks rather than full experiment runs.
- [x] Baseline autoencoder and local PatchCore training commands use the shared preprocessing pipeline.
- [ ] Run syntax/import checks on a clean environment.
- [ ] Execute the notebook cells in order on the target runtime and fix any runtime-specific issues.
- [ ] Check every final report/table against the primary experiment artifacts.

## Shared conventions

- Input images: float32, NCHW, RGB, range [0, 1] for RGB pipelines; PCA uses grayscale tensors where configured.
- Model fitting uses defect-free training images only.
- Test labels and masks are reserved for final evaluation.
- Larger anomaly-map values always mean more anomalous.
