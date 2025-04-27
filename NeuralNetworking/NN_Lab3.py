# Import dependencies
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# device configuration
device = torch.device('cuda' if torch.cuda .is_available() else 'cpu')

# hyper-parameters
input_size = 784  # 28*28
hidden_siez = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Download and load the MNIST dataset
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform = transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform = transforms.ToTensor(),
    download=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False,
)
samples = iter(test_loader)
example_data, example_targets = next(samples)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(example_data[i][0], cmap='plasma')
plt.show()
