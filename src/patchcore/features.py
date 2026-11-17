"""Pretrained CNN patch feature extraction from ImageNet-normalized RGB tensors."""

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureExtractor:
    def __init__(self, pretrained=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        self.model = wide_resnet50_2(weights=weights).to(self.device).eval()
        self._features = {}
        self.model.layer2.register_forward_hook(self._hook("layer2"))
        self.model.layer3.register_forward_hook(self._hook("layer3"))

    def _hook(self, name):
        def save(_module, _inputs, output):
            self._features[name] = output
        return save

    @torch.no_grad()
    def __call__(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected ImageNet-normalized RGB images with shape N x 3 x H x W.")
        if not torch.is_floating_point(images):
            raise TypeError("Expected floating-point image tensors.")
        self._features.clear()
        self.model(images.to(self.device))
        f2, f3 = self._features["layer2"], self._features["layer3"]
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)
