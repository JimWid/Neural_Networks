# Neural Networks From Scratch (Numpy and Pytorch)
This repository is a personal learning journey through the fundamentals of neural networks. The goal is to **understand deep learning not just as a user of libraries, but as someone who can derive, implement and understand every component from first principles**.

There is Numpy and Pytorch implementations, living side by side intentionally: this way we can see a more raw version using Numpy and then a more production grade implementation using Pytorch. I believe this is great for learning :)

---

## Repository Structure

```
Neural_Networks/
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
## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repo
git clone https://github.com/JimWid/Neural_Networks.git
cd Neural_Networks

# Install dependencies
uv sync

# Run an experiment (example)
uv run testing/regression.py
```
---

## Acknowledgements

The base code and conceptual structure for the NumPy implementation were inspired by the excellent work of **The Independent Code**, a YouTube channel dedicated to building machine learning systems from mathematical first principles.
- YouTube: [Neural Network from Scratch | Mathematics & Python Code](https://youtube.com/playlist?list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm)
- GitHub: [TheIndependentCode/Neural-Network](https://github.com/TheIndependentCode/Neural-Network)

Their clear derivations of backpropagation and modular layer design were instrumental in shaping this project. Highly recommended for anyone who wants to genuinely understand what's happening inside a neural network.
