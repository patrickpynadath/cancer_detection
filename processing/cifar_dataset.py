import os
import os.path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

IDX_TO_LABEL = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
ANIMAL_CLASS_IDX = [2, 3, 4, 5, 6]
OBJ_CLASS_IDX = [1, 7, 8, 9]

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


def get_values(dataset):
    new_values = []
    for i in range(len(dataset)):
        if dataset[i][1] in ANIMAL_CLASS_IDX:
            new_values.append(1)
        else:
            new_values.append(0)
    return new_values




