'''
Optimizing Model Parameters
Now that we have a model and data it's time to train, validate and test our model by
optimizing its parameters on our data. Training a model is an iterative process;
in each iteration the model makes a guess about the output, calculates error in its
guess (loss), collects the derivatives of the error with respect to its parameters
(as seen in part 5) and optimizes these parameters using gradient descent.
'''

# load data from part 2 (datasets/dataloaders) & 4 (build model)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

'''
Hyperparameters
Hyperparameters are adjustable parameters that let you
control the model optimization process.
Different hyperparameter values can impact model training and convergence rates.

We define the following hyperparameters for training:
- epochs: the number of times to iterate over the dataset
- batch size: the number of data samples propagated through the network
  before parameters are updated
- learning rate: how much to update model parameters at each batch/epoch;
  smaller values yield slower learning speed,
  higher values may result in unpredictable behaviour during training
'''

learning_rate = 1e-3
batch_size = 64
epochs = 5

'''
Optimization Loop
Once hyperparameters are set, models can be trained and optimized
using an optimization loop. Each iteration of the loop is called an epoch.

Each epoch consists of two parts:
- training loop: iterate over training dataset,
  trying to converge to optimal parameters
- validation/test loop: iterate over test dataset,
  checking if model performance is improving


Loss Function
When presented with training data, our untrained network is likely to
give incorrect answers. The loss function measures the 
degree of dissimilarity of obtained results to the target value.
We are trying to minimize the loss function during training.
To calculate loss, we make a prediction based on our input data sample
and compare against the true data label value.

Common loss functions include "nn.MSELoss" (mean square error) for regression tasks
and "nn.NLLLoss" (negative log likelihood) for classification.
"nn.CrossEntropyLoss" combines "nn.LogSoftmax" and "nn.NLLLoss"

We pass our model's output logits to "nn.CrossEntropyLoss",
which normalizes the logits and computes prediction error.
'''
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

