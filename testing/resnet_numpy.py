import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from numpy_nn.layers import Dense
from numpy_nn.reshape import Reshape
from numpy_nn.residual import ResidualBlock
from numpy_nn.convolutional import Convolution
from numpy_nn.activations import ReLU, BatchNormalization, Softmax

from network_numpy import test, train, load, save
from numpy_nn.losses import categorical_cross_entropy, categorical_cross_entropy_prime

from keras.datasets import mnist

# Loading MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Only 0.5%
train_size = int(0.02 * len(X_train))
test_size = int(0.02 * len(X_test))

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

# RestNet Architecture!
network = [
    Convolution((1, 28, 28), 3, 8, padding=1),
    BatchNormalization(),
    ReLU(),
    ResidualBlock((8, 28, 28), 3, 8, epsilon=1e-8, padding=1),
    Reshape((8, 28, 28), (8 * 28 * 28, 1)),
    Dense(8 * 28 * 28, 10),
    Softmax()
]

# Model Path
path = "models/resnet_best.pkl"
epochs = 25
lr = 0.01

# Train if model doesn't exist
if os.path.exists(path):
    network = load(path)
else:
    train(network, categorical_cross_entropy, categorical_cross_entropy_prime, X_train, y_train, epochs, lr)
    save(network, path)

# Testing saved model
test(network, X_test, y_test, n=10)