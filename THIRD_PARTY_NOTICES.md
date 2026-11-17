# Third-party software, datasets and research references

The MIT license in the repository root applies only to original project code and original documentation authored by the project contributors. It does not relicense datasets, pretrained weights, papers, or third-party packages.

## Dataset

**MVTec AD** is provided by MVTec Software GmbH under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0). Commercial use is not permitted. Users must review and comply with the license, provide attribution, and apply the ShareAlike condition to adaptations of the dataset. Obtain the dataset from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The dataset is not included in this repository.

Suggested dataset citation:

Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

## Principal software dependencies

The project depends on third-party packages including PyTorch, Torchvision, NumPy, SciPy, scikit-learn, scikit-image, OpenCV, Matplotlib, pandas, Pillow, PyYAML and tqdm. Their licenses are supplied by their respective maintainers and distributions. In particular:

- [PyTorch](https://github.com/pytorch/pytorch) — BSD-style license; bundled components may have additional licenses.
- [Torchvision](https://github.com/pytorch/vision) — BSD-3-Clause.
- [Anomalib](https://github.com/open-edge-platform/anomalib) — Apache License 2.0. The completed PatchCore benchmark notebook uses Anomalib 2.7.0.
- Other dependencies — consult the license files and metadata of the exact installed versions. Transitive dependencies may carry additional notices.

Do not redistribute package source, pretrained weights, or third-party assets from this repository unless their respective terms permit it. The original license and notices for every redistributed component must be retained.

## Pretrained weights

The Anomalib PatchCore experiment uses ImageNet-pretrained Wide ResNet-50-2 features. The weights are obtained through the Torchvision/Anomalib software stack at runtime and are not committed here. Review the terms that apply to the exact weights and their source before redistributing them.

## ImageNet-pretrained backbone weights

The completed PatchCore benchmark uses an ImageNet-pretrained Wide ResNet-50-2 backbone through Anomalib. The exact weight file/revision and checksum from the historical run are not retained in the repository, and the benchmark notebook does not commit the weights.

TorchVision warns that pretrained weights can have their own licenses or terms derived from the training dataset and that the user is responsible for checking permission for the intended use. The ImageNet access terms state that the database is for non-commercial research and educational use. Review the applicable ImageNet access agreement and any weight-specific terms before redistributing weights or using them outside the permitted research/educational context. Do not assume the repository's MIT license applies to pretrained weights.

- [TorchVision pretrained-weight guidance](https://docs.pytorch.org/vision/master/models.html)
- [ImageNet terms of access](https://www.image-net.org/download)
- [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE)
- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)

## Research methods

- PatchCore: Roth, K. et al. (2022). *Towards Total Recall in Industrial Anomaly Detection*. CVPR.
- MemAE: Gong, D. et al. (2019). *Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection*. ICCV, pp. 1705–1714. DOI: 10.1109/ICCV.2019.00179.

Research papers are cited as scholarly references; the citations do not grant permission to reproduce their figures, text, or supplementary materials.
