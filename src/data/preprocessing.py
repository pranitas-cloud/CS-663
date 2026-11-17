import numpy as np
import torch
import torch.nn.functional as F


def prepare_image(image, size=256):
    array = np.asarray(image)
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
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255.0)
    if tensor.shape[0] == 3:
        weights = tensor.new_tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
        tensor = (tensor * weights).sum(dim=0, keepdim=True)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return tensor.squeeze(0).numpy().astype(np.float32)


def prepare_mask(mask, size=256, mode="max"):
    if mask is None:
        return np.zeros((size, size), dtype=bool)
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array.max(axis=2)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Expected a square single-channel mask.")
    tensor = torch.from_numpy(np.ascontiguousarray(array > 0)).float()[None, None]
    if tensor.shape[-1] != size:
        if mode == "max":
            if size > tensor.shape[-1]:
                raise ValueError("Max mask resizing only supports downsampling.")
            tensor = F.adaptive_max_pool2d(tensor, (size, size))
        elif mode == "nearest":
            tensor = F.interpolate(tensor, size=(size, size), mode="nearest")
        else:
            raise ValueError("Mask mode must be max or nearest.")
    return tensor.squeeze().numpy().astype(bool)
