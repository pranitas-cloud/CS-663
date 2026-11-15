"""Pretrained CNN patch feature extraction (feature backbone only)."""
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
        self._features.clear()
        x = images.to(self.device)
        # ImageNet normalization is required for pretrained weights.
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.model((x - mean) / std)
        f2, f3 = self._features["layer2"], self._features["layer3"]
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)
