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
    dataset.data = dataset.data.astype(np.float32)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = torch.tensor(dataset.data[i] / 255, dtype=torch.float32 ).to(device)
            # Image data usually comes as HWC; EEG tensors are already channel-first
            if sample.ndim == 3 and sample.shape[-1] in (1, 3):
                sample = sample.permute(2, 0, 1)
            d = lock.transform_sample(sample, dataset.targets[i]).clamp(0,1)
            if d.ndim == 3 and d.shape[0] in (1, 3):
                d = d.permute(1, 2, 0)
            d = d.detach().cpu().numpy()
            d = d*255
            d = d.astype(np.uint8)
            dataset.data[i] = d
    dataset.data = dataset.data.astype(np.uint8)
    return dataset

"""
    Input: 1) Pytorch dataset 2) learnability lock
    Output: data loader
"""
def gen_unlearnable_dataset_targeted(dataset, lock, target):
    assert target != None
    dataset.data = dataset.data.astype(np.float32)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            if dataset.targets[i] not in target: continue
            sample = torch.tensor(dataset.data[i] / 255, dtype=torch.float32 ).to(device)
            if sample.ndim == 3 and sample.shape[-1] in (1, 3):
                sample = sample.permute(2, 0, 1)
            d = lock.transform_sample(sample, dataset.targets[i]).clamp(0,1)
            if d.ndim == 3 and d.shape[0] in (1, 3):
                d = d.permute(1, 2, 0)
            d = d.detach().cpu().numpy()
            d = d*255
            d = d.astype(np.uint8)
            dataset.data[i] = d
    dataset.data = dataset.data.astype(np.uint8)
    return dataset

def retrive_clean_dataset(dataset, lock):
    dataset.data = dataset.data.astype(np.float32)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = torch.tensor(dataset.data[i] / 255, dtype=torch.float32 ).to(device)
            if sample.ndim == 3 and sample.shape[-1] in (1, 3):
                sample = sample.permute(2, 0, 1)
            d = lock.inv_transform_sample(sample, dataset.targets[i]).clamp(0,1)
            if d.ndim == 3 and d.shape[0] in (1, 3):
                d = d.permute(1, 2, 0)
            d = d.detach().cpu().numpy()
            d = d*255
            d = d.astype(np.uint8)
            dataset.data[i] = d
    dataset.data = dataset.data.astype(np.uint8)
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