# Final code and licensing audit

## Audit scope and conclusion

This review checked the tracked repository inventory, dependency declarations, model-loading paths, experiment notebooks, notices, data-handling rules and committed result assets.

**Repository-level status: generally clean, with two external confirmations still required.**

- The root MIT license is limited to original contributor-authored project code and documentation.
- MVTec AD data is documented separately under CC BY-NC-SA 4.0 and is not committed.
- Anomalib, PyTorch and TorchVision are consumed as dependencies; their source trees and pretrained weights are not vendored into this repository.
- The tracked repository inventory contains no raw MVTec image/mask files, model checkpoint binaries, pretrained weight files or committed qualitative image figures.
- The source tree has no separate vendor/third-party implementation directory.
- The PatchCore benchmark uses Anomalib 2.7.0 and is clearly distinguished from the independent local random-memory-bank prototype.
- The exact pretrained-weight checkpoint/revision used in the historical PatchCore run is not recorded. Weight-specific and ImageNet access terms therefore remain an external verification item.
- A contributor-by-contributor confirmation that no unlicensed code, figures or other assets were copied into the project is still required; a repository inventory cannot establish authorship on its own.

## 1. Original project code

The root [MIT License](../LICENSE) applies only to original code and documentation authored by project contributors. It does not grant rights to MVTec AD, ImageNet, pretrained model weights, third-party software or research-paper figures.

The contributors previously approved MIT licensing for their original project work. Each contributor should still confirm that the files they contributed are original or have compatible attribution/licensing.

## 2. Dataset

**MVTec AD** is released by MVTec Software GmbH under CC BY-NC-SA 4.0. The terms require attribution, restrict commercial use and apply ShareAlike conditions to adaptations. The official page states these terms and provides the dataset and evaluation-code information.

Repository checks:
- The raw MVTec dataset is not tracked.
- `.gitignore` excludes contents under `data/` while retaining `data/README.md`.
- Dataset terms and a suggested citation are recorded in `data/README.md` and `THIRD_PARTY_NOTICES.md`.
- The result/plot directories currently contain placeholders, not MVTec image figures.

**Figure handling:** qualitative plots generated from MVTec inputs/masks are not automatically covered by the repository MIT license. Keep them outside Git unless their distribution is permitted under the dataset terms and the attribution/licensing conditions are met.

Official source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

## 3. Third-party software

| Component | How it is used | License/notice status |
|---|---|---|
| PyTorch | Model construction, training and tensor operations | BSD-style license in upstream repository; notice linked in `THIRD_PARTY_NOTICES.md` |
| TorchVision | Wide ResNet-50-2 architecture and pretrained model interface | BSD-3-Clause; pretrained weights have separate weight/data terms to check |
| Anomalib 2.7.0 | Implementation used for the reported PatchCore benchmark | Apache License 2.0; dependency is pinned in `requirements-anomalib.txt` |
| NumPy, SciPy, scikit-learn, scikit-image, OpenCV, Matplotlib, pandas, Pillow, PyYAML, tqdm | Numerical, image-processing, evaluation and plotting dependencies | Third-party dependencies, not copied into the repository; exact-version/transitive notices should be checked before redistribution |

The repository does not vendor Anomalib or third-party package source. Its dependency files and CI lock identify package versions for reproducibility; they are not a complete legal license inventory of every transitive package.

Upstream references:
- PyTorch: https://github.com/pytorch/pytorch
- TorchVision: https://github.com/pytorch/vision
- Anomalib: https://github.com/open-edge-platform/anomalib
- Anomalib 2.7.0 release: https://github.com/open-edge-platform/anomalib/releases/tag/2.7.0

## 4. Pretrained Wide ResNet-50-2 weights and ImageNet

The Anomalib PatchCore benchmark requests an ImageNet-pretrained `wide_resnet50_2` backbone. The historical run record identifies the architecture and says pretrained weights were used, but does not retain the exact weight file/revision, source URL or checksum.

TorchVision's model documentation warns that pretrained weights may have licenses or terms derived from their training datasets and that users are responsible for determining permission for their intended use. The ImageNet terms of access restrict use of the database to non-commercial research and educational purposes.

Audit decision:
- The code obtains weights at runtime; no weight files are committed.
- The historical weight variant and SHA-256 are not recorded.
- Before distributing a checkpoint or using it outside academic/non-commercial research or education, verify the applicable ImageNet agreement and the exact weight-specific terms.
- Do not claim that MIT covers the pretrained checkpoint.

References:
- TorchVision pretrained-weight guidance: https://docs.pytorch.org/vision/master/models.html
- ImageNet access terms: https://www.image-net.org/download
- TorchVision license: https://github.com/pytorch/vision/blob/main/LICENSE
- PyTorch license: https://github.com/pytorch/pytorch/blob/main/LICENSE

## 5. Research papers and figures

The repository cites the MVTec AD, PatchCore and MemAE papers in `docs/references.md`. The code uses the published method ideas as research references; the citations do not grant permission to reproduce paper figures, text or supplementary assets.

No third-party paper figures or raster image assets were found in the tracked repository inventory. If a report uses a paper figure, obtain permission where required or redraw an original explanatory diagram and cite the underlying method.

## 6. Code provenance review

The tracked repository contains project modules under `src/data/`, `src/autoencoder/`, `src/memae/`, `src/patchcore/` and `src/evaluation/`. There is no vendored package tree or separate third-party source directory. The local PatchCore-style module implements a random-subset memory bank and is explicitly not represented as the Anomalib PatchCore benchmark.

This structural review cannot prove that every line was independently authored. All contributors must check their own contributions for copied snippets, adapted code, borrowed diagrams and generated figures, and add source/license notices where applicable.

## 7. Repository hygiene

- Dataset folders and generated experiment/model files are ignored by Git.
- Lightweight CSV/JSON outputs under `results/tables/` and top-level `results/*.json` manifests are now allowed to be committed; large outputs remain ignored.
- No raw MVTec images, masks, checkpoints or qualitative image figures are part of the current tracked inventory.
- No external package source is vendored.

## 8. Outstanding confirmations

- [ ] Every contributor confirms the provenance and license of their contributed code and figures.
- [ ] Verify the exact historical Wide ResNet-50-2 weight variant/source/checksum and applicable terms, or preserve a fresh manifest on a reproduction run.
- [ ] Confirm that the MVTec dataset was obtained and used under its non-commercial research/educational terms.
- [ ] Review exact-version/transitive package licenses before redistributing an environment or binary bundle.
- [ ] Review report/presentation figures individually before sharing them outside the course/team.

This audit is a project-level source and notice review, not formal legal advice.
