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
- [x] Run Python compilation and all 21 tests successfully on a fresh Ubuntu/Python 3.11 GitHub Actions runner (latest validation: [run #80](https://github.com/pranitas-cloud/CS-663/actions/runs/44111815614)).
- [ ] Run each full Kaggle notebook from a fresh session and record the resulting Kaggle execution URL.
- [x] Confirm from the notebook control flow that MemAE training/evaluation does not require a Google Drive secret; Drive backup is optional.
- [x] Generate and commit the resolved Ubuntu/Python 3.11 CPU lock; CI installs from that lock.

## Results and reporting

- [x] Completed PatchCore/Anomalib and MemAE result tables committed.
- [x] Macro and per-category tables linked from the README.
- [x] Local PatchCore-style prototype distinguished from the completed Anomalib benchmark.
- [x] PatchCore macro metrics in the record notebook reconciled with the committed macro table.
- [ ] Verify reported metrics against original run artifacts and freeze the final report tables.
- [x] Add runtime environment-manifest generation to both full Kaggle notebooks.
- [ ] Run fresh Kaggle sessions and preserve the generated manifests with the final experiment artifacts.

## Code and notebooks

- [x] Global PCA notebook rewritten with explicit mean-centering, sample-space eigendecomposition, projection, reconstruction and residual scoring.
- [x] Patch PCA notebook rewritten with explicit patch sampling, covariance eigendecomposition, reconstruction and overlap-averaged residual maps.
- [ ] Execute both rewritten PCA notebooks on MVTec data and verify their category-level result files.
- [x] All 12 notebooks have valid, unique cell IDs and pass notebook JSON validation.
- [x] Notebook outputs cleared and execution counters reset.
- [x] Code-cell comments and stale saved output removed; explanatory documentation retained in Markdown/docstrings.
- [x] Old account-specific Drive and Kaggle dataset paths removed from notebook code.
- [x] Former notebook placeholders labelled as smoke checks rather than full experiment runs.
- [x] Baseline autoencoder and local PatchCore training commands use the shared preprocessing pipeline.
- [x] Run Python compileall and model/data smoke tests in the locked CI environment.
- [ ] Execute both full Kaggle notebooks from fresh GPU sessions and fix any runtime-specific issues.
- [ ] Check every final report/table against the primary experiment artifacts.

## Shared conventions

- Input images: float32, NCHW, RGB, range [0, 1] for RGB pipelines; PCA uses grayscale tensors where configured.
- Model fitting uses defect-free training images only.
- Test labels and masks are reserved for final evaluation.
- Larger anomaly-map values always mean more anomalous.
