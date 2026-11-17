# Final results comparison and interpretation

## Scope and reporting rule

The primary methods in this project are PatchCore, MemAE, Global PCA and Patch-wise PCA. The convolutional autoencoder is retained as a secondary training baseline, not promoted as a completed benchmark.

The committed category tables contain completed 15-category results for PatchCore (Anomalib 2.7.0) and MemAE. The rewritten PCA notebooks and the convolutional autoencoder do not yet have committed MVTec evaluation tables. Their metrics are therefore marked as not run/not available in the consolidated CSV; no values are estimated or inferred.

Source tables:
- `results/tables/patchcore_anomalib_categories.csv`
- `results/tables/patchcore_anomalib_macro.csv`
- `results/tables/memae_categories.csv`
- `results/tables/memae_macro.csv`

The machine-readable cross-method table is `results/tables/method_comparison.csv`.

## Cross-method macro comparison

All displayed macro values are arithmetic means of the 15 category-level values in the committed result tables.

| Method | Image AUROC | Image AP | Image F1 | Pixel AUROC | Pixel AP | Pixel F1 | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore (Anomalib 2.7.0) | **0.982848** | Not retained | **0.971479** | **0.980105** | Not retained | **0.580585** | 38.99 min |
| MemAE | 0.672232 | 0.841480 | 0.425354 | 0.738080 | 0.144815 | 0.125190 | 54.97 min |
| Global PCA | Not run | Not run | Not run | Not run | Not run | Not run | Not recorded |
| Patch-wise PCA | Not run | Not run | Not run | Not run | Not run | Not run | Not recorded |
| Convolutional autoencoder | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not recorded |

PatchCore AP is explicitly unavailable because it was not present in the retained PatchCore summary. It is not valid to substitute an AP value from another run or derive it from AUROC/F1.

## Main findings

### PatchCore is the strongest completed benchmark

The Anomalib PatchCore run has the strongest recorded ranking performance: image AUROC 0.982848 and pixel AUROC 0.980105. It uses ImageNet-pretrained Wide ResNet-50-2 intermediate features from `layer2` and `layer3`, a 10% coreset memory bank, and 9 nearest neighbours. The feature representation is useful because it compares local patterns in a pretrained visual feature space rather than relying only on raw pixel reconstruction.

The result is strong across most categories. Image AUROC is at least 0.97 for 13 of the 15 categories. The lowest image AUROC values are toothbrush (0.916667), pill (0.947081), and screw (0.970691). The weakest pixel AUROC values are wood (0.931554), tile (0.955723), and transistor (0.973172); the other categories are above 0.98.

Pixel F1 (0.580585 macro) is noticeably lower than pixel AUROC (0.980105). This is not contradictory: AUROC evaluates ranking over all thresholds, whereas F1 evaluates a single binary threshold. A strong ranking can still yield a modest F1 when the chosen threshold does not provide a good precision/recall balance, particularly with sparse defect pixels. The retained summary does not preserve enough threshold detail to diagnose the PatchCore F1 value further.

The category-level pixel F1 results are weakest for screw (0.368145), grid (0.384241), leather (0.440368), and wood (0.468301). These are useful categories to inspect in qualitative maps, but the aggregate CSV alone cannot identify the exact defect subtype or spatial reason for each error.

### MemAE is competitive on some categories but less consistent

MemAE's macro file records image AUROC 0.672232, image AP 0.841480, image F1 0.425354, pixel AUROC 0.738080, pixel AP 0.144815 and pixel F1 0.125190. These are the recorded macro values; the category CSV does not currently support a complete independent reconstruction of all six values. In particular, the image AP macro has no per-category image AP column, and the F1 fields in the category CSV are malformed/ambiguous (see the artifact audit). The AP distinction matters because AP is sensitive to the proportion of positive examples, and MVTec categories have different test-set class proportions. AUROC and AP should therefore be read together rather than treated as interchangeable measures.

MemAE is highly category-dependent. Its strongest image AUROC values are wood (0.977193), hazelnut (0.944643), and tile (0.919913). The weakest are metal_nut (0.339198), cable (0.382684), carpet (0.400883), and transistor (0.437500). On pixel AUROC, hazelnut (0.928628) and screw (0.892525) are among its strongest, while transistor (0.548030), cable (0.582605), and carpet (0.602848) are among its weakest.

The committed per-category pixel AP values are low for grid (0.024678), carpet (0.026838), screw (0.028190), and cable (0.037837). This indicates that high anomaly scores are not consistently concentrated on defective pixels for these categories. Since anomaly pixels occupy a small fraction of an image, false-positive pixels can strongly affect precision.

MemAE is reconstruction-based: the encoder, memory module and decoder learn to reproduce normal training appearances. A likely limitation is that a sufficiently expressive reconstruction path can also reproduce some anomalous structures, reducing the residual contrast between normal and defective regions. Categories with complex textures, reflective surfaces, variable geometry or subtle/local defects can also be difficult for this formulation. These are plausible explanations for the observed category-level pattern, not confirmed defect-by-defect diagnoses: the original anomaly maps and per-image artifacts were not accessible during this repository audit.

### F1 values and category-table limitations

The MemAE notebook explicitly calibrates the image threshold at the 99th percentile and the pixel threshold at the 99.5th percentile of reconstruction errors from held-out normal training images. Its image score is the mean of the largest 1% of reconstruction-error pixels.

The committed MemAE category CSV has a structural discrepancy: its original header declared both `image_F1Score` and `pixel_F1Score`, but each data row contained only one F1 value. The table has been normalized to retain that value under the neutral field name `f1_score_field_unverified`; it is not assigned to either image F1 or pixel F1. Its mean (0.125190) matches the macro file's pixel F1, but that alone is not enough to assert its identity. The macro file's image F1 (0.425354) is preserved as recorded but cannot be recomputed from the current category CSV.

The PatchCore summary identifies the Anomalib evaluator but does not retain the exact F1 threshold values or calibration protocol. Consequently, PatchCore and MemAE F1 scores are reported faithfully from their respective macro files, but they are not a controlled head-to-head F1 comparison. The AUROC values are more suitable for a broad ranking comparison, while AP comparisons are currently impossible for PatchCore because its AP values were not retained.

The runtime values are also run records, not a strict speed benchmark: PatchCore took 38.99 minutes total (about 2.60 minutes/category), and MemAE took 54.97 minutes total (about 3.66 minutes/category). Exact hardware and environment evidence is incomplete, so small runtime differences should not be over-interpreted.

## Why the PCA methods remain important

Global PCA and Patch-wise PCA provide interpretable, classical baselines. Global PCA models the distribution of entire images and may therefore be sensitive to object alignment and broad appearance variation. Patch-wise PCA models local texture/structure and is more naturally aligned with local defect localization, but its performance depends on patch size, stride, sampling and the normal patch distribution.

Neither PCA notebook has yet been executed on the MVTec test split after the rewrite. No claim that either PCA method outperforms or underperforms PatchCore/MemAE is made here. Once run, report both category tables and compare using the same image resolution, test split and metric definitions.

## Convolutional autoencoder status

The convolutional autoencoder is not a placeholder: `src/autoencoder/model.py` contains a three-stage encoder-decoder, `src/autoencoder/train.py` implements normal-only MSE training and checkpoint saving, and `tests/test_models.py` checks its forward output shape, range and finite values.

However, the trainer does not currently implement a complete anomaly-detection evaluation pipeline: it does not produce anomaly maps, calibrate thresholds, calculate image/pixel metrics, or aggregate category results. No trained checkpoint or benchmark result is committed. It is therefore retained as an implemented secondary training baseline, but it is not presented as a completed fourth method or a headline result.

## Recommended qualitative inspection

For the final report, inspect representative true positives, false positives and false negatives from the retained PatchCore and MemAE outputs, especially:
- PatchCore: screw, grid, wood and toothbrush.
- MemAE: cable, carpet, metal_nut, transistor and the strong wood/hazelnut cases.

The metric tables identify where to look; the underlying image/mask/map artifacts are needed to substantiate visual explanations. Do not claim a particular defect morphology caused a failure until those maps have been inspected.

## Reproducibility and provenance caveats

- PatchCore: recorded Anomalib 2.7.0, PyTorch 2.11.0+cu130, GPU count 1, and the model/training settings listed in the experiment record. Exact Python/platform/driver manifest from the original run was not retained in Git.
- MemAE: configuration, category metrics, thresholds, per-image records and anomaly maps are designed to be written by the full notebook. The original environment manifest and raw category artifacts are not committed in this repository.
- The current full notebooks include environment-manifest generation, but that does not prove the original historical run emitted those manifests.
- The cross-method CSV records missing values as blank rather than inventing results.

## Conclusion

Among the two completed 15-category benchmarks, PatchCore is the strongest method in the retained image- and pixel-level AUROC results. MemAE provides a useful reconstruction/memory-based comparison, but its performance varies substantially across categories and its pixel AP/F1 are low. PCA remains an important classical baseline and must be run before its comparative performance can be discussed. The convolutional autoencoder is kept as a secondary training baseline until an evaluation pipeline and results exist.
