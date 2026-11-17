from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from src.data.dataset import MVTecDataset


class PreparedDataset(Dataset):
    def __init__(self, raw, image_size=256, mask_mode="max"):
        self.raw = raw
        self.samples = raw.samples
        self.image_size = int(image_size)
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        sample = self.raw[index]
        array = np.asarray(sample["image"])
        if array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {array.dtype}.")
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
            raise ValueError(f"Unsupported image shape {array.shape}.")
        if array.shape[0] != array.shape[1]:
            raise ValueError("MVTec images are expected to be square.")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        image = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255.0)
        if image.shape[0] == 3:
            weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
            image = (image * weights).sum(dim=0, keepdim=True)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        mask_array = sample["mask"]
        if mask_array is None:
            mask = torch.zeros(1, self.image_size, self.image_size)
        else:
            mask_array = np.asarray(mask_array)
            if mask_array.ndim == 3:
                mask_array = mask_array.max(axis=2)
            if mask_array.ndim != 2 or mask_array.shape[0] != mask_array.shape[1]:
                raise ValueError("Expected a square single-channel mask.")
            mask = torch.from_numpy(np.ascontiguousarray(mask_array > 0)).float()[None, None]
            if mask.shape[-1] != self.image_size:
                if self.mask_mode == "max":
                    mask = F.adaptive_max_pool2d(mask, (self.image_size, self.image_size))
                elif self.mask_mode == "nearest":
                    mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="nearest")
                else:
                    raise ValueError("Mask mode must be max or nearest.")
            mask = mask.squeeze(0)

        return {
            "image": image,
            "mask": mask,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "image_path": sample["image_path"],
            "orig_size": torch.tensor(sample["image"].shape[:2]),
        }


def load_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dataset(config, method, split, category=None):
    if method != "pca":
        raise ValueError("This repository contains the Global PCA pipeline only.")
    data = config["data"]
    raw = MVTecDataset(
        Path(data["root"]),
        category or data["category"],
        split=split,
        val_fraction=data.get("val_fraction", 0.10),
        seed=config.get("seed", 42),
    )
    return PreparedDataset(
        raw,
        image_size=config.get("pca", {}).get("image_size", data.get("image_size", 256)),
        mask_mode=data.get("mask_resize", "max"),
    )


def stack_images(dataset):
    return torch.stack([dataset[index]["image"] for index in range(len(dataset))])
