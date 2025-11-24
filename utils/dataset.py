"""
Module: Data loading utilities
Author: hust-marx2
Date: 2023/9/22
Description: Provides Dataset class and DataLoader creation function with robustness and flexibility.
"""

import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


# 随机种子初始化,保证相同的种子实验可重复性
def set_seed(myseed=23721):
    """Sets random seeds for reproducibility."""
    # torch.set_num_threads(1) # Consider removing for performance if strict determinism isn't required
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)  # For multi-GPU
    np.random.seed(myseed)
    os.environ["PYTHONHASHSEED"] = str(myseed)  # Note: typo corrected from "PYTHONSEED"
    torch.backends.cudnn.enabled = False  # Ensures deterministic algorithms (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disables benchmarking for deterministic behavior


class MyDataset(Dataset):
    """
    A simple PyTorch Dataset for loading data and labels.
    Assumes input data and labels are numpy arrays.
    """

    def __init__(self, data, label, include_index: bool = False):
        """
        Initializes the dataset.
        Args:
            data (np.ndarray): The input data array.
            label (np.ndarray): The corresponding label array.
            include_index (bool): Whether to return the sample index for each item.
        """
        if data.shape[0] != label.shape[0]:
            raise ValueError(
                f"Data and label must have the same length. Got {data.shape[0]} and {label.shape[0]}."
            )
        self.data = data
        self.label = label
        self.include_index = include_index

    def __len__(self):
        """Returns the total number of samples."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Fetches a single sample and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the data sample (np.float32), the label, and
            optionally the sample index when ``include_index`` is True.
        """
        data = self.data[idx].astype(np.float32)
        label = self.label[idx]
        if self.include_index:
            return data, label, idx
        return data, label


class GPUReadyDataset(Dataset):
    """
    Dataset variant that returns pre-converted tensors (optionally on GPU) to
    minimize host-to-device copies for latency-sensitive training such as
    LLock.
    """

    def __init__(self, data, label, include_index: bool = False):
        if data.shape[0] != label.shape[0]:
            raise ValueError(
                f"Data and label must have the same length. Got {data.shape[0]} and {label.shape[0]}."
            )
        self.data = data
        self.label = label
        self.include_index = include_index

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        if self.include_index:
            return data, label, idx
        return data, label


def ToDataLoader(
    data,
    label,
    mode,
    batch_size=32,
    shuffle=None,
    num_workers=0,
    pin_memory=True,
    drop_last=None,
    include_index: bool = False,
):
    """
    Creates a PyTorch DataLoader for the given data and labels.

    Args:
        data (np.ndarray): The input data array.
        label (np.ndarray): The corresponding label array.
        mode (str): 'train' or 'test'.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True for 'train', False for 'test'.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.
        pin_memory (bool, optional): Whether to use pinned memory for faster GPU transfer. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True for 'train', False for 'test'.

    Returns:
        DataLoader: The configured PyTorch DataLoader.

    Raises:
        ValueError: If the mode is not 'train' or 'test'.
    """
    if shuffle is None:
        shuffle = mode == "train"
    if drop_last is None:
        drop_last = mode == "train"  # Usually True for training, False for testing

    if mode not in ["train", "test"]:
        raise ValueError(f"Mode must be 'train' or 'test'. Got '{mode}'.")

    dataset = MyDataset(data, label, include_index=include_index)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataloader


def ToDataLoaderLLock(
    data,
    label,
    mode,
    batch_size=32,
    shuffle=None,
    num_workers=0,
    pin_memory=False,
    drop_last=None,
    include_index: bool = False,
):
    """
    DataLoader variant optimized for LLock GPU-first training. It assumes the
    provided data/labels are already torch.Tensors (ideally on GPU or pinned
    memory) and avoids additional conversions/copies.
    """
    if shuffle is None:
        shuffle = mode == "train"
    if drop_last is None:
        drop_last = mode == "train"
    if mode not in ["train", "test"]:
        raise ValueError(f"Mode must be 'train' or 'test'. Got '{mode}'.")

    dataset = GPUReadyDataset(data, label, include_index=include_index)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataloader

# --- Example Usage ---
# set_seed(23721) # Call at the start of your script
# train_loader = get_dataloader(train_data, train_labels, mode="train", batch_size=128)
# test_loader = get_dataloader(test_data, test_labels, mode="test", batch_size=128)
# for batch_data, batch_labels in train_loader:
#     # Your training logic here
#     pass
