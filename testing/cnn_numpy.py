import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from numpy_nn.convolutional import Convolution
from numpy_nn.activations import ReLU
from numpy_nn.reshape import Reshape
from numpy_nn.activations import Sigmoid
from numpy_nn.layers import Dense
from network import test, train, load, save
from numpy_nn.losses import mse, mse_prime

from keras.datasets import mnist

# Loading MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Only 20%
train_size = int(0.2 * len(X_train))
test_size = int(0.2 * len(X_test))

X_train, y_train = X_train[:train_size], y_train[:train_size]
X_test, y_test = X_test[:test_size], y_test[:test_size]

# Normalization (1 channel, 28x28 pixels)
X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28) / 255.0

# One-hot encoding of labels
def one_hot(y, num_classes=10):
    encoded = np.zeros((len(y), num_classes, 1))
    for i, label in enumerate(y):
        encoded[i][label] = 1

    return encoded

y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Model / Network architecture
network = [
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

# Model Path
path = "models/cnn_best.pkl"
epochs = 25
lr = 0.01

# Train if model doesn't exist
if os.path.exists(path):
    network = load(path)
else:
    train(network, mse, mse_prime, X_train, y_train, epochs, lr)
    save(network, path)

# Testing saved model
test(network, X_test, y_test, n=100)
