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
- learning rate: how much to update model parameters at each batch/epoch;
  smaller values yield slower learning speed,
  higher values may result in unpredictable behaviour during training
- batch size: the number of data samples propagated through the network
  before parameters are updated
- epochs: the number of times to iterate over the dataset
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

'''
Optimizer
Optimization is the process of adjusting model parameters to reduce model error
in each training step. Opimization algorithms define how this process is performed
(in this example we use Stochastic Gradient Descent). All optimization logic
is encapsulated in the "optimizer" object. We use the SGD but there are many others
available in PyTorch including ADAM and RMSProp that work better for different
datasets and models.
'''

# initialize optimizer registering model parameters to train and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
In the training loop, optimization happens in 3 steps:
- Call optimizer.zero_grad() to reset the gradients of model parameters.
  Gradients by default add up; to prevent double-counting,
  we explicitly zero them at each iteration.
- Backpropagate prediction loss calling loss.backward(),
  PyTorch deposits the gradients of loss with respect to each parameter.
- Once we have our gradients, we call optimizer.step() to adjust the parameters by
  the gradients collected in the backward pass.
'''

'''
Full Implementation
We define "train_loop" which loops over our optimization code,
and "test_loop" that evaluates the model's performance against our test data.
'''


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Initialize loss function/optimizer then pass to train/test loops.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")