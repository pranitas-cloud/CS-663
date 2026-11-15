"""Train the convolutional autoencoder using only defect-free images."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import MVTecDataset
from src.autoencoder.model import ConvAutoencoder


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = MVTecDataset(args.data_root, args.category, "train", args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    model = ConvAutoencoder(latent_channels=args.latent_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x = batch["image"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), x)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        print(f"epoch={epoch+1} train_mse={total/len(ds):.6f}")
    out = Path("experiments/autoencoder"); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "category": args.category, "image_size": args.image_size}, out / f"{args.category}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    p.add_argument("--category", default="bottle")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-channels", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())
