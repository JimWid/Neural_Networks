import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pytorch_nn.convolution import Convolution
from network_pytorch import train, save, load

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

# Converting to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Datasets
train_dataset = TensorDataset(X_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_test, y_test)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Model
model = Convolution(
    input_size=(28, 28),
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    stride=1,
    num_classes=10,
    activation=nn.ReLU(), # Note: Do NOT use Sigmoid, use ReLU :) Pytorch works better with it!
    padding=0,
)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Path
path = "models/cnn_pytorch_best.pkl"

# Train if model doesn't exist
if os.path.exists(path):
    model = load(path)
else:
    train(model, loss, optimizer, train_data_loader, val_data_loader, epochs=10, device=device)
    save(model, path)
