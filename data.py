# data.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, num_workers=4, img_size=224, dataset='cifar10'):
    """
    Get data loaders for the specified dataset.
    Args:
        dataset: str, either 'cifar10' or 'fashion_mnist'
    """
    # Data transformations
    if dataset.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:  # fashion_mnist
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # Load datasets based on choice
    if dataset.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
    elif dataset.lower() == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset must be either 'cifar10' or 'fashion_mnist'")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
