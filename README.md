# Neural Networks From Scratch (Numpy and Pytorch)
This repository is a personal learning journey through the fundamentals of neural networks. The goal is to **understand deep learning not just as a user of libraries, but as someone who can derive, implement and understand every component from first principles**.

There is Numpy and Pytorch implementations, living side by side intentionally: this way we can see a more raw version using Numpy and then a more production grade implementation using Pytorch. I believe this is great for learning :)

---

## Repository Structure

```
Neural_Networks/
│
├── datasets/           # Datasets classes for models
│
├── numpy_nn/           # Implementation using only Numpy
│
├── pytorch_nn/         # Same implementation using PyTorch
│
├── testing/            # Trying both Numpy and Pytorch models
│
├── network_numpy.py    # Helper functions (train, validation, save/load model) for Numpy
├── network_pytorch.py  # Helper functions for Pytorch
├── pyproject.toml      # Project dependencies and environment config
└── uv.lock             # Locked dependency versions (UV)
```

---

## Implemented
These are the models that have been implemented so far:
- Linear Regression
- Convolution
- ResNet
- YOLO
  
---

## Numpy Implementation Example
The numpy implementation usually goes by **implementing the components from scratch**, whether is an new **type of layer, activation function or loss**. Each one of them with their **forward** and **backward** functions.

```python
# RelU Implementation
class ReLU(Layer): # Rectified Linear Unit (ReLU)
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)
```
Then we create a testing file to create the **network**. As simple as it can be, **basically a list of all the layers and activations!**

```python
# Convolution Architecture!
network = [
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    MaxPooling(stride=2),
    Reshape((5, 13, 13), (5 * 13 * 13, 1)),
    Dense(5 * 13 * 13, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]
```
**This is our model, yay!**

## Pytorch Implementation Example
With Pytorch we **skip** the forward and backward implemention at all, since **Pytorch will handle that for us internally**, we usually **create a class for the model/network we want and its forward.**

```python
class Convolution(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride, num_classes, activation, padding=0):
        super().__init__()
        
        self.height, self.width = input_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        out_h = (self.height + 2 * padding - kernel_size) // stride + 1
        out_w = (self.width + 2 * padding - kernel_size) // stride + 1
        
        self.flat = nn.Flatten()
        self.activation = activation    
        self.linear_1 = nn.Linear(out_channels * out_h * out_w, 100)
        self.linear_2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.flat(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x
```
**This is the exact same network as the Numpy implementation:**
- ``self.conv`` = ``Convolution()``
- ``self.linear`` = ``Dense()``
- ``self.flat`` = ``Reshape()``
- ``self.activation`` = ``RelU()`` or ``Sigmoid()`` or any activation
  
>  Notice how the ``forward()`` is very similar to the numpy network

We then create a testing file calling this class as our model.

```python
# Convolution Model
model = Convolution(
    input_size=(28, 28),
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    stride=1,
    num_classes=10,
    activation=nn.ReLU(),
    padding=0,
)
```
**And that's our model!**

---

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repo
git clone https://github.com/JimWid/Neural_Networks.git
cd Neural_Networks

# Install dependencies
uv sync

# Run an experiment (example)
uv run testing/regression_numpy.py
```
---

## Acknowledgements

The base code and conceptual structure for the NumPy implementation were inspired by the excellent work of **The Independent Code**, a YouTube channel dedicated to building machine learning systems from mathematical first principles.
- YouTube: [Neural Network from Scratch | Mathematics & Python Code](https://youtube.com/playlist?list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm)
- GitHub: [TheIndependentCode/Neural-Network](https://github.com/TheIndependentCode/Neural-Network)

Their clear derivations of backpropagation and modular layer design were instrumental in shaping this project. Highly recommended for anyone who wants to genuinely understand what's happening inside a neural network.
