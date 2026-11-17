import torch

from src.autoencoder.model import ConvAutoencoder
from src.patchcore.memory_bank import MemoryBank
from src.memae.model import MemoryAutoencoder, memae_loss


def test_convolutional_autoencoder_forward_shape():
    model = ConvAutoencoder(in_channels=3, latent_channels=16).eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        reconstruction = model(images)
    assert reconstruction.shape == images.shape
    assert torch.isfinite(reconstruction).all()
    assert 0.0 <= reconstruction.min()
    assert reconstruction.max() <= 1.0


def test_memae_forward_and_loss_are_finite():
    model = MemoryAutoencoder(memory_size=8).eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        reconstruction, entropy = model(images)
        total_loss, reconstruction_loss = memae_loss(
            reconstruction,
            images,
            entropy,
        )
    assert reconstruction.shape == images.shape
    assert torch.isfinite(reconstruction).all()
    assert torch.isfinite(total_loss)
    assert torch.isfinite(reconstruction_loss)


def test_random_memory_bank_fit_and_query():
    feature_batch = torch.rand(2, 8, 4, 4)
    bank = MemoryBank(max_samples=10).fit([feature_batch])
    scores = bank.query(feature_batch)
    assert bank.features.shape == (10, 8)
    assert scores.shape == (2, 16)
    assert torch.isfinite(scores).all()
