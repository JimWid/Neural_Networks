import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from .convolutional import Convolution
from .reshape import Reshape
from activations import Sigmoid, Softmax, ReLU
from layers import Dense
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from network import train, predict

train_data = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Prepare data
X_train = np.array([img.numpy().reshape(1, 28, 28) for img, _ in train_data])
y_train = np.array([np.eye(10)[label].reshape(10, 1) for _, label in train_data])

X_test = np.array([img.numpy().reshape(1, 28, 28) for img, _ in test_data])
y_test = np.array([np.eye(10)[label].reshape(10, 1) for _, label in test_data])

subset_size = 2000
train_indices = np.random.choice(len(X_train), subset_size, replace=False)

X_train = X_train[train_indices]
y_train = y_train[train_indices]

test_subset_size = 400
test_indices = np.random.choice(len(X_test), test_subset_size, replace=False)
X_test = X_test[test_indices]
y_test = y_test[test_indices]

# Build the neural network
network = [
    Convolution((1, 28, 28), 3, 5),
    ReLU(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    ReLU(),
    Dense(100, 10),
    Softmax()
]

# Train the network
train(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    X_train,
    y_train,
    epochs=50,
    learning_rate=0.01
)

# Evaluate the network
for x, y in zip(X_test, y_test):
    output = predict(network, x)
    print("Predicted:", np.argmax(output), "\tTrue:", np.argmax(y))

