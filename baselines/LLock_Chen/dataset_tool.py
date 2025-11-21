from torchvision import datasets, transforms
import torch 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


device = 'cuda'

"""
    Input: 1) Pytorch dataset 2) learnability lock
    Output: data loader
"""
def gen_unlearnable_dataset(dataset, lock):
    dataset.data = dataset.data.astype(np.float32)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            d = torch.tensor(dataset.data[i] / 255, dtype=torch.float32 ).to(device).permute(2,0,1)
            d = lock.transform_sample(d, dataset.targets[i]).clamp(0,1)
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
            d = torch.tensor(dataset.data[i] / 255, dtype=torch.float32 ).to(device).permute(2,0,1)
            d = lock.transform_sample(d, dataset.targets[i]).clamp(0,1)
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
            d = torch.tensor(dataset.data[i] / 255, dtype=torch.float32 ).to(device).permute(2,0,1)
            d = lock.inv_transform_sample(d, dataset.targets[i]).clamp(0,1)
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