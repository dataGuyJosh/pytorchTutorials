'''
Datasets & Data Loaders
Code for processing data samples can get messy and hard to maintain;
we ideally want our dataset code to be decoupled from our model training code
for better readability and modularity.

PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset
that allow you to use pre-loaded datasets as well as your own data.
Dataset stores the samples and their corresponding labels,
and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST)
that subclass torch.utils.data.Dataset and implement functions specific to the particular data.
They can be used to prototype and benchmark your model.
'''

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)