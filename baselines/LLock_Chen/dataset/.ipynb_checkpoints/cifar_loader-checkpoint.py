
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_cifar10(batch_size=256):
    train_transform = [
    #             transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    test_transform = [
    #             transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)


    clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=batch_size,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False)
    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False)
    return clean_train_loader, clean_test_loader

def load_cifar10_data():
    train_transform = [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    test_transform = [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)


    return clean_train_dataset, clean_test_dataset




def load_cifar100(batch_size=256):
    train_transform = [
    #             transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    test_transform = [
    #             transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    clean_train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR100(root='../datasets', train=False, download=True, transform=test_transform)


    clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=batch_size,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False)
    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False)
    return clean_train_loader, clean_test_loader


def load_cifar100_data(batch_size=256):
    train_transform = [
    #             transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    test_transform = [
    #             transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                         (0.2023, 0.1994, 0.2010))
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    clean_train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR100(root='../datasets', train=False, download=True, transform=test_transform)


    return clean_train_dataset, clean_test_dataset