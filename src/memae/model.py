"""Memory-augmented autoencoder for unsupervised anomaly detection."""

import torch
import torch.nn.functional as F
from torch import nn


class MemoryModule(nn.Module):
    """Sparse memory retrieval over encoded spatial feature vectors."""

    def __init__(
        self,
        channels: int = 128,
        memory_size: int = 256,
        shrink_threshold: float = 1e-3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.memory = nn.Parameter(torch.empty(memory_size, channels))
        nn.init.xavier_uniform_(self.memory)
        self.shrink_threshold = shrink_threshold
        self.temperature = temperature

    def forward(self, feature_map: torch.Tensor):
        batch, channels, height, width = feature_map.shape
        queries = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
        queries = F.normalize(queries, dim=1)
        memory = F.normalize(self.memory, dim=1)

        similarities = queries @ memory.T
        soft_attention = torch.softmax(similarities / self.temperature, dim=1)
        sparse_attention = F.relu(soft_attention - self.shrink_threshold)
        denominator = sparse_attention.sum(dim=1, keepdim=True)

        fallback = F.one_hot(
            soft_attention.argmax(dim=1), num_classes=soft_attention.shape[1]
        ).to(dtype=soft_attention.dtype)

        sparse_attention = torch.where(
            denominator > 1e-12,
            sparse_attention / denominator.clamp_min(1e-12),
            fallback,
        )

        retrieved = sparse_attention @ self.memory
        entropy = -(
            soft_attention * torch.log(soft_attention.clamp_min(1e-12))
        ).sum(dim=1).mean()

        retrieved = retrieved.reshape(batch, height, width, channels)
        retrieved = retrieved.permute(0, 3, 1, 2)
        return retrieved, entropy


class MemoryAutoencoder(nn.Module):
    """Encoder -> sparse memory -> decoder."""

    def __init__(
        self,
        memory_size: int = 256,
        shrink_threshold: float = 1e-3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.memory = MemoryModule(
            channels=128,
            memory_size=memory_size,
            shrink_threshold=shrink_threshold,
            temperature=temperature,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor):
        encoded = self.encoder(images)
        memory_features, entropy = self.memory(encoded)
        reconstruction = self.decoder(memory_features)
        return reconstruction, entropy


def memae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    entropy: torch.Tensor,
    entropy_weight: float = 1e-4,
):
    reconstruction_loss = F.mse_loss(reconstruction, target)
    total_loss = reconstruction_loss + entropy_weight * entropy
    return total_loss, reconstruction_loss
