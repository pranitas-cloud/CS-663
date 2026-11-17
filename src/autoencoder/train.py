"""Train the convolutional autoencoder using only preprocessed normal images."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.autoencoder.model import ConvAutoencoder
from src.data.dataset import MVTecDataset
from src.data.pipelines import AutoencoderPreprocessor, PreparedDataset


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    raw_dataset = MVTecDataset(
        args.data_root,
        args.category,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    dataset = PreparedDataset(
        raw_dataset,
        AutoencoderPreprocessor(image_size=args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = ConvAutoencoder(latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]

        mean_loss = total_loss / len(dataset)
        print(f"epoch={epoch + 1} train_mse={mean_loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.cpu().state_dict(),
        "category": args.category,
        "image_size": args.image_size,
        "latent_channels": args.latent_channels,
        "seed": args.seed,
    }
    output_path = output_dir / f"{args.category}.pt"
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", default="experiments/autoencoder")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
