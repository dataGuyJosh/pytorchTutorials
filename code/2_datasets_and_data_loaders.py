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

# iterating & visualizing datasets
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# creating a custom dataset
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    # initialize the directory containing the images, the annotations file, and both transforms
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    # the number of samples in the dataset
    def __len__(self):
        return len(self.img_labels)
    # load and return a sample from the dataset at given index
    def __getitem__(self, idx):
        # identifies the image’s location on disk
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # converts to a tensor
        image = read_image(img_path)
        # retrieves the corresponding label from the csv data in self.img_labels
        label = self.img_labels.iloc[idx, 1]
        # calls the transform functions on them (if applicable)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # returns the tensor image and corresponding label in a tuple
        return image, label

# preparing data for training
'''
The "Dataset" retrieves our dataset's features and labels one sample at a time.
While training a model, we typically want to pass samples in "minibatches",
reshuffle the data at every epoch to reduce model overfitting,
and use Python's multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.
'''
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# iterate through DataLoader --> display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")