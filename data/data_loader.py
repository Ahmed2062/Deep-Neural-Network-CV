"""MNIST data loaders. The dataset is downloaded on first run (not committed)."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Mean and standard deviation of MNIST training pixels.
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_data_loaders(data_dir="data", batch_size=128, seed=42, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0, 255] -> [0, 1]
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader
