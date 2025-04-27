# dependency: torch, torchvision, matplotlib
# practical seciton of the course computational neuroscience
import torch
from  torchvision.transforms import ToTensor
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn

# Loading the Fashion MNIST dataset
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)
test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

# Preparing the data for training and testing with DataLoader
from torch.utils.data import DataLoader
train = DataLoader(training_data, batch_size=64, shuffle=True)
test = DataLoader(test_data, batch_size=64, shuffle=True)

# Building the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = nn.flatten(x)
        result = self.linear_relu_stack(x)
        return result
    

# Get a batch of training data
data_iter = iter(train)
images, labels = next(data_iter)

# Plot a few images from the dataset
fig, axes = plt.subplots(1, 5, figsize=(12, 6))  # Display 5 images
for i in range(5):
    image = images[i].squeeze()  # Remove extra dimensions
    axes[i].imshow(image, cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"Label: {labels[i].item()}")

plt.show()