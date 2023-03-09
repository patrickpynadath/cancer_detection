import os
import os.path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar_sets():
    root_dir = r'../'
    # transformations from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/datasets.py
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                           download=True, transform=test_transform)

    return trainset, testset


def get_cifar_dataloaders(batch_size):
    trainset, testset = get_cifar_sets()
    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


