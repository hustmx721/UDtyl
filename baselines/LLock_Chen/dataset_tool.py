from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


# Pick GPU when available but gracefully fall back to CPU so EEG tensors can be preprocessed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    Input: 1) Pytorch dataset 2) learnability lock
    Output: data loader
"""
def gen_unlearnable_dataset(dataset, lock):
    original_dtype = dataset.data.dtype
    is_image_like = dataset.data.ndim >= 3 and dataset.data.shape[-1] in (1, 3)
    dataset.data = dataset.data.astype(np.float32)

    def _to_channel_first(tensor):
        if tensor.ndim == 3 and is_image_like:
            return tensor.permute(2, 0, 1)
        return tensor

    def _to_original_layout(tensor):
        if tensor.ndim == 3 and is_image_like:
            return tensor.permute(1, 2, 0)
        return tensor

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = torch.tensor(dataset.data[i], dtype=torch.float32).to(device)
            # Normalize image inputs but preserve EEG amplitudes
            if is_image_like:
                sample = sample / 255.0
            sample = _to_channel_first(sample)

            d = lock.transform_sample(sample, dataset.targets[i]).clamp(0, 1)
            d = _to_original_layout(d)

            d = d.detach().cpu().numpy()
            if is_image_like:
                d = (d * 255).astype(np.uint8)
            else:
                d = d.astype(original_dtype, copy=False)
            dataset.data[i] = d

    # Preserve original dtype for non-image data, default to uint8 for images
    dataset.data = dataset.data.astype(np.uint8 if is_image_like else original_dtype)
    return dataset

"""
    Input: 1) Pytorch dataset 2) learnability lock
    Output: data loader
"""
def gen_unlearnable_dataset_targeted(dataset, lock, target):
    assert target is not None
    original_dtype = dataset.data.dtype
    is_image_like = dataset.data.ndim >= 3 and dataset.data.shape[-1] in (1, 3)
    dataset.data = dataset.data.astype(np.float32)

    def _to_channel_first(tensor):
        if tensor.ndim == 3 and is_image_like:
            return tensor.permute(2, 0, 1)
        return tensor

    def _to_original_layout(tensor):
        if tensor.ndim == 3 and is_image_like:
            return tensor.permute(1, 2, 0)
        return tensor

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            if dataset.targets[i] not in target:
                continue
            sample = torch.tensor(dataset.data[i], dtype=torch.float32).to(device)
            if is_image_like:
                sample = sample / 255.0
            sample = _to_channel_first(sample)
            d = lock.transform_sample(sample, dataset.targets[i]).clamp(0, 1)
            d = _to_original_layout(d)
            d = d.detach().cpu().numpy()
            if is_image_like:
                d = (d * 255).astype(np.uint8)
            else:
                d = d.astype(original_dtype, copy=False)
            dataset.data[i] = d
    dataset.data = dataset.data.astype(np.uint8 if is_image_like else original_dtype)
    return dataset

def retrive_clean_dataset(dataset, lock):
    original_dtype = dataset.data.dtype
    is_image_like = dataset.data.ndim >= 3 and dataset.data.shape[-1] in (1, 3)
    dataset.data = dataset.data.astype(np.float32)

    def _to_channel_first(tensor):
        if tensor.ndim == 3 and is_image_like:
            return tensor.permute(2, 0, 1)
        return tensor

    def _to_original_layout(tensor):
        if tensor.ndim == 3 and is_image_like:
            return tensor.permute(1, 2, 0)
        return tensor

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = torch.tensor(dataset.data[i], dtype=torch.float32).to(device)
            if is_image_like:
                sample = sample / 255.0
            sample = _to_channel_first(sample)
            d = lock.inv_transform_sample(sample, dataset.targets[i]).clamp(0, 1)
            d = _to_original_layout(d)
            d = d.detach().cpu().numpy()
            if is_image_like:
                d = (d * 255).astype(np.uint8)
            else:
                d = d.astype(original_dtype, copy=False)
            dataset.data[i] = d
    dataset.data = dataset.data.astype(np.uint8 if is_image_like else original_dtype)
    return dataset


def linear_transform_sample(W, b, x, label):
    # element-wise/pixel-wise transform
    result = W[label] * x + b[label]
#         result = result * 1/(1+self.epsilon)
    return result


def inv_linear_transform_sample(W, b, x, label):
#         x = x * (1 + self.epsilon)
    result = (x - b[label]) * 1/W[label]
    return result
