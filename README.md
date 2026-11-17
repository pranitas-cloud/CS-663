# Industrial Anomaly Detection

**CS 663 — Digital Image Processing | Course project**
AUTHORS:
Aniruddha Deore [24B2182], Pranita Bodkhe[24B2144]
This repository studies unsupervised industrial anomaly detection and pixel-level defect localization on the MVTec Anomaly Detection (MVTec AD) benchmark.

## Methods and experiment tracks

The repository contains both project implementations and completed benchmark notebooks. These are related, but not all use the same implementation:

| Track | Implementation | Status |
|---|---|---|
| Global PCA | `notebooks/02_pca_experiments.ipynb` | Implementation ready; 15-category results pending |
| Patch PCA | `notebooks/02b_patch_pca_experiments.ipynb` | Implementation ready; 15-category results pending |
| Convolutional autoencoder | `src/autoencoder/` | Trainer implemented; no full benchmark results recorded |
| PatchCore-style prototype | `src/patchcore/` | Prototype only; random memory-bank sampling, not the reported benchmark |
| PatchCore benchmark | Anomalib 2.7.0 in the full Kaggle notebook | Completed 15-category run |
| MemAE | Project implementation in `src/memae/` and full Kaggle notebook | Completed 15-category run |

**Important implementation distinction:** the reported PatchCore benchmark in the results tables was run using Anomalib 2.7.0. The separate `src/patchcore/` prototype is not an exact reproduction of Anomalib PatchCore: its current memory-bank sampler is random rather than greedy coreset selection. Do not attribute the Anomalib benchmark metrics to the local prototype.

The PCA baselines are implemented directly in the two notebooks using NumPy eigenvalue decomposition; no prebuilt PCA estimator is used. Scikit-learn is used only for evaluation metrics. The project evaluates image-level anomaly detection and pixel-level localization. The experiment protocol uses defect-free training images; official test images and masks are held back for evaluation.

## Repository layout

```text
configs/                 Experiment configuration files
data/                    Dataset instructions; raw MVTec data is not committed
docs/                    Methodology, results, references and reproduction notes
experiments/              Local run outputs (ignored by Git)
notebooks/                Exploration, baseline, and full benchmark notebooks
results/                  Committed result tables; generated artifacts are ignored
scripts/                  Dataset validation utilities
src/data/                 MVTec loading and preprocessing
src/autoencoder/          Convolutional autoencoder model and trainer
src/patchcore/            Local PatchCore-style prototype and memory-bank trainer
src/memae/                Memory-augmented autoencoder model
src/evaluation/           Evaluation metrics and visualization helpers
src/analysis/             Defect-property and failure analysis
tests/                    Synthetic-data unit tests
```

## 1. Environment setup

Use Python 3.11 for the tested CI workflow. Python 3.10+ is supported by the package metadata.

```bash
git clone https://github.com/pranitas-cloud/CS-663.git
cd CS-663
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pytorch-cpu.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

For Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For an exact clean install on Ubuntu x86_64 with Python 3.11, use the resolved lock generated from the passing GitHub Actions run:

```bash
python -m pip install -r requirements-lock-ubuntu-py311.txt
python -m pip install --no-deps -e .
```

That lock includes PyTorch 2.11.0+cpu and Torchvision 0.26.0+cpu, with all other resolved packages pinned to the versions used by CI. It is specific to the matching Linux/Python environment; do not use it as a macOS, Windows, or CUDA lock. The CI job regenerates the lock artifact when dependencies change. For other platforms, use the appropriate PyTorch wheel set and the version constraints in `requirements.txt`.

For the completed Anomalib PatchCore benchmark (PyTorch 2.11.0 + CUDA 13.0), install the matching framework wheels first:

```bash
python -m pip install -r requirements-pytorch-cu130.txt
python -m pip install -r requirements-anomalib.txt
```

The Anomalib dependency file pins Anomalib 2.7.0, PyTorch 2.11.0 and Torchvision 0.26.0. These versions satisfy Anomalib 2.7.0's minimum framework requirements. GPU driver, CUDA runtime, wheel availability and hardware compatibility remain machine-specific. The full Kaggle notebook also installs Anomalib 2.7.0 in its setup cell. Do not install Anomalib into the baseline CPU environment unless running that benchmark.

## 2. Obtain and validate MVTec AD

1. Download **MVTec AD** (not MVTec AD 2) from the [official MVTec dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
2. Review the dataset's CC BY-NC-SA 4.0 terms before use or redistribution.
3. Extract it so the directory directly contains the 15 category folders.
4. Place the extracted folder at `data/mvtec_anomaly_detection/`, or provide its location to the validator/notebook.

Expected structure:

```text
data/mvtec_anomaly_detection/
  bottle/
    train/good/
    test/good/
    test/<defect_type>/
    ground_truth/<defect_type>/
  cable/
  capsule/
  ...
  zipper/
```

Validate the entire dataset:

```bash
python scripts/validate_dataset.py --data-root data/mvtec_anomaly_detection
```

Validate a single category:

```bash
python scripts/validate_dataset.py --data-root data/mvtec_anomaly_detection --category bottle
```

The validator checks the expected category folders, readable square PNG images, and the presence and dimensions of defect masks. It does not download the dataset or replace the official evaluation code.

**Do not commit MVTec images, masks, archives, or derived visualizations containing dataset imagery.** The dataset is separately licensed and is not covered by this repository's MIT license. The repository's `.gitignore` excludes local data and experiment artifacts.

See [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset terms and attribution.

## 3. Run tests

The tests create small synthetic MVTec-style folders; they do not require the real MVTec dataset or a GPU.

```bash
python -m pytest -q
```

Or use the Makefile:

```bash
make test
make validate-data
```

GitHub Actions runs the same CPU test suite on pushes to `main`, pull requests, and manual dispatches. A green CI run validates the tested code paths; it does not mean that the 15-category GPU benchmarks have been re-run.

## 4. Run the repository baseline trainers

The global and patch-wise PCA implementations are written directly in `notebooks/02_pca_experiments.ipynb` and `notebooks/02b_patch_pca_experiments.ipynb` using NumPy eigenvalue decomposition. They do not use a prebuilt PCA estimator. Both notebooks default to a one-category bottle smoke test. Set `RUN_ALL_CATEGORIES = True` in the setup cell to run all 15 MVTec categories and save per-category and macro metric tables under `results/tables/`; fitted models are saved under `experiments/pca/`.

The convolutional autoencoder trainer uses the shared image preprocessing pipeline and only the normal training split:

```bash
python -m src.autoencoder.train --data-root data/mvtec_anomaly_detection --category bottle --epochs 2
```

The local PatchCore-style prototype builds a random-subset feature memory bank from normal training images and saves it under `experiments/patchcore/`:

```bash
python -m src.patchcore.detector --data-root data/mvtec_anomaly_detection --category bottle --max-samples 5000
```

The local PatchCore-style command is a feature-memory training step, not a full Anomalib benchmark reproduction or an end-to-end evaluation command. It may download the ImageNet-pretrained Wide ResNet-50-2 weights on first use. The full reported PatchCore benchmark is reproduced with the Anomalib Kaggle notebook below.

## 5. Reproduce the full benchmark runs

The two completed 15-category runs are supplied as self-contained Kaggle notebooks:

- [PatchCore — Anomalib 2.7.0 full run](notebooks/04_patchcore_kaggle_full_run.ipynb)
- [MemAE — full training and evaluation run](notebooks/06_memae_kaggle_full_run.ipynb)

To reproduce them:

1. Open the relevant notebook in Kaggle.
2. Attach a copy of MVTec AD that you obtained in accordance with its license. The notebook searches Kaggle input mounts for the dataset folder or archive; check the dataset-discovery cell if your mount uses a nonstandard layout.
3. Select a GPU accelerator for practical full-run execution.
4. Run the notebook from the first cell. The PatchCore notebook runs a one-category pilot before proceeding through all categories.
5. Keep the generated outputs from `/kaggle/working`; download or back them up before the Kaggle session ends.

The MemAE notebook's Google Drive backup is optional. Training and evaluation run locally in the Kaggle session without a `GDRIVE_TOKEN`; Drive is only an optional backup/resume destination.

The full notebooks record their hyperparameters, category-level metrics, thresholds, runtime and visualizations. Each full-run notebook also writes an `environment.json` manifest into its output folder, capturing Python, PyTorch, Torchvision, CUDA, GPU and relevant package versions. Exact runtime and numerical reproducibility can vary with GPU model, library build, and nondeterministic accelerator kernels. Use the committed result tables as the original run record, not as a promise that every machine will reproduce every decimal.

## 6. Reported results

### PatchCore — Anomalib 2.7.0

The completed run covered all 15 categories. It used ImageNet-pretrained `wide_resnet50_2`, feature layers `layer2` and `layer3`, coreset sampling ratio 0.10, 9 nearest neighbours, train/evaluation batch sizes of 8, 0 data-loader workers, and seed 42. The recorded total runtime was 38.99 minutes.

Macro results:

| Metric | Macro score |
|---|---:|
| Image AUROC | 0.982848 |
| Image F1 | 0.971479 |
| Pixel AUROC | 0.980105 |
| Pixel F1 | 0.580585 |

Full data: [category metrics](results/tables/patchcore_anomalib_categories.csv) · [macro metrics](results/tables/patchcore_anomalib_macro.csv)

### MemAE — project implementation

The completed run trained one model per category using normal training images, with a held-out normal validation subset. Inputs were resized to 256×256. The recorded setup used memory size 256, hard-shrinkage threshold 1e-3, temperature 0.07, maximum 40 epochs, early-stopping patience 8, batch size 16, learning rate 1e-3, weight decay 1e-5, and seed 42. The total measured category runtime was approximately 54.97 minutes.

| Metric | Macro score |
|---|---:|
| Image AUROC | 0.672232 |
| Image AP | 0.841480 |
| Image F1 | 0.425354 |
| Pixel AUROC | 0.738080 |
| Pixel AP | 0.144815 |
| Pixel F1 | 0.125190 |

Full data: [category metrics](results/tables/memae_categories.csv) · [macro metrics](results/tables/memae_macro.csv)

F1 thresholds and evaluation implementations were not established to be identical across the PatchCore and MemAE runs. Interpret F1 comparisons cautiously. The post-hoc MemAE image-score ablation is exploratory and is not the primary reported result.

See [docs/completed_experiments_nov2026.md](docs/completed_experiments_nov2026.md) for the experiment record and [docs/methodology.md](docs/methodology.md) for methodology details.

## 7. Notebooks

- [01 — Dataset exploration](notebooks/01_dataset_exploration.ipynb)
- [01b — Fast preprocessing audit](notebooks/01b_preprocessing_audit_fast.ipynb)
- [02 — PCA experiments](notebooks/02_pca_experiments.ipynb)
- [02b — Patch PCA experiments](notebooks/02b_patch_pca_experiments.ipynb)
- [03 — Autoencoder model smoke check](notebooks/03_autoencoder_experiments.ipynb)
- [04 — Local PatchCore-style prototype smoke check](notebooks/04_patchcore_experiments.ipynb)
- [05 — PatchCore Anomalib run record](notebooks/05_patchcore_anomalib_record.ipynb)
- [06 — MemAE experiment record](notebooks/06_memae_experiment_record.ipynb)

The record notebooks are concise summaries. The two full Kaggle notebooks linked above are the execution artifacts for the completed Anomalib PatchCore and MemAE runs.

## 8. Licensing and citation

- Original project code and documentation: [MIT License](LICENSE).
- MVTec AD dataset: separate CC BY-NC-SA 4.0 terms; not included or relicensed here.
- Third-party packages and pretrained weights: retain and follow their respective licenses and notices.
- Research methods: cite the original papers listed in [docs/references.md](docs/references.md).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse or redistribution.
