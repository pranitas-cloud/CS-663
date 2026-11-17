# Original experiment artifact audit

## Audit result

The committed repository preserves the aggregate and per-category metric tables for the completed PatchCore and MemAE runs. The notebook code documents the additional run artifacts that should have been written to Kaggle output folders and, for MemAE, optionally backed up to Google Drive.

**The original Google Drive files were not independently opened or checksum-verified during this audit.** This document distinguishes files confirmed in Git from artifacts that are documented by the experiment notebooks but still need an owner-side Drive check. A file-writing statement in a notebook is not proof that a particular historical file survived the original Kaggle session.

## 1. Artifacts confirmed in Git

| Method | File | Contents | Rows |
|---|---|---|---:|
| PatchCore | `results/tables/patchcore_anomalib_categories.csv` | Per-category runtime, image AUROC/F1, pixel AUROC/F1 | 15 |
| PatchCore | `results/tables/patchcore_anomalib_macro.csv` | Four retained macro metrics | 4 |
| MemAE | `results/tables/memae_categories.csv` | Per-category runtime, test counts, image AUROC, pixel AUROC/AP, one F1 field with unresolved image/pixel meaning | 15 |
| MemAE | `results/tables/memae_macro.csv` | Six macro metrics | 6 |

The metric tables are committed source records. Their current Git blob identifiers are included in `results/experiment_manifest.json` so that later changes can be distinguished from the audited versions.

### Metric-table discrepancies to resolve

The MemAE macro table includes `image_AP = 0.841480`, but the committed 15-row MemAE category CSV does not contain an `image_AP` column. The macro value is retained as recorded, but it cannot be independently recomputed from the category CSV alone.

The MemAE category CSV also had a 10-column header but only 9 data fields per row. The original pair of labels `image_F1Score` and `pixel_F1Score` could not be safely assigned to the single available F1 value. The table was normalized to one neutral column, `f1_score_field_unverified`, without changing the numeric row values. The mean of this field is 0.125190, which matches the macro file's pixel F1, but that numerical match is not sufficient evidence to relabel the field. The macro image F1 (0.425354) and pixel F1 (0.125190) are retained as reported in the macro file; the original category JSONs or notebook output are required to reconcile the category-level F1 scores.

PatchCore's retained category and macro tables do not contain image AP or pixel AP. Both are reported as unavailable; no values are imputed.

## 2. PatchCore original run artifacts

The experiment record says model checkpoints, category predictions and visualisation folders were retained in Google Drive rather than Git. The current full-run notebook writes to:

```text
/kaggle/working/patchcore_runs_anomalib_2_7_0/
```

The notebook is designed to create:
- `environment.json`
- One `test_results.json` per category
- Anomalib checkpoints and category outputs
- `all_categories_results.json`
- `completed_categories.json`
- `all_categories_metrics.csv`
- `aggregate_summary.csv`
- Failure records, where applicable
- `PatchCore_MVTec_Kaggle_results.zip`

The exact historical Google Drive folder is not stated in the committed experiment record. The expected Kaggle output path above is from the current notebook's code and must not be mistaken for proof that the historical ZIP is present in Drive.

### Environment evidence

The retained experiment notes record:
- Anomalib 2.7.0
- PyTorch 2.11.0+cu130
- One GPU

The original Python version, operating platform, Torchvision version, exact GPU model, CUDA runtime/driver and full package list are not preserved in the Git experiment notes. The current full notebook writes those details to `environment.json`; the presence of that file from the historical run is not verified.

## 3. MemAE original run artifacts

The experiment record documents the Drive location:

```text
gdrive:CS663Project/memae_runs
```

The full notebook writes locally to:

```text
/kaggle/working/memae_runs/
```

Its output logic is designed to save:
- `environment.json`
- `categories/<category>/best_model.pt`
- `categories/<category>/training_history.csv`
- `categories/<category>/metrics.json`
- `categories/<category>/calibration.json`
- `categories/<category>/per_image_predictions.csv`
- `categories/<category>/anomaly_maps.npz` (when raw-map saving is enabled)
- `visualizations/<category>_examples.png`
- `metrics/all_categories_metrics.csv`
- `metrics/aggregate_summary.csv`
- `metrics/progress.json`
- `metrics/failures.json`, when failures are recorded

### Environment evidence

The current full notebook pins PyTorch 2.11.0 and Torchvision 0.26.0 using the CUDA 13.0 wheel index, and writes the actual Python/platform/CUDA/GPU/package details to `environment.json` during execution. Those notebook settings are not a substitute for the historical run's environment snapshot. The exact historical MemAE runtime manifest is not present in Git and was not independently checked in Drive.

## 4. What is and is not verified

| Item | Git status | External artifact status |
|---|---|---|
| PatchCore per-category/macro CSVs | Confirmed | Not needed to read the committed summary |
| MemAE per-category/macro CSVs | Confirmed | Per-category image AP source still needs reconciliation |
| PatchCore checkpoints/maps/predictions | Not committed | Reported in Drive; not independently verified |
| MemAE checkpoints/maps/predictions/visualisations | Not committed | Documented Drive path; not independently verified |
| Historical PatchCore environment manifest | Not committed | Current notebook can generate one; historical copy unverified |
| Historical MemAE environment manifest | Not committed | Current notebook can generate one; historical copy unverified |
| MVTec raw images/masks | Correctly excluded | Dataset must remain separately licensed |

## 5. Preservation checklist for the original Drive exports

When the run folders are available, preserve one immutable archive per method and keep a small manifest alongside it:

1. Copy the full PatchCore run output and the MemAE run folder without renaming the per-category files.
2. Confirm all 15 category records exist and are marked completed.
3. Confirm the category-level CSV row count is 15 and that its macro values match the committed macro file.
4. Preserve each original `environment.json`, model configuration, thresholds, per-image predictions and anomaly maps.
5. Generate SHA-256 checksums for the exported ZIP and the key CSV/JSON files. Add those checksums to the experiment manifest.
6. Reconcile MemAE `image_AP` and both F1 fields at category level from the original JSON results so the macro values can be reproduced.
7. Keep MVTec image/mask data and image-containing visualisations out of Git unless the dataset terms permit the intended redistribution and the necessary attribution/licensing conditions are satisfied.

## 6. Convolutional autoencoder

The autoencoder is not a completed experiment artifact. The repository has a model, a normal-only training script and a forward-pass test, but no committed trained checkpoint, test predictions, anomaly maps or metric table. It is retained as a secondary training baseline and excluded from completed benchmark claims.
