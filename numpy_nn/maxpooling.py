import numpy as np
from numpy_nn.layers import Layer

class MaxPooling(Layer):
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, input):
        self.input = input
        depth, height, width = input.shape

        output_h = height // self.stride
        output_w = width // self.stride
        self.output = np.zeros((depth, output_h, output_w))

        for d in range(depth):
            for i in range(output_h):
                for j in range(output_w):
                    region = input[
                        d,
                        i * self.stride: (i + 1) * self.stride,
                        j * self.stride: (j + 1) * self.stride
                    ]

                    self.output[d, i, j] = np.max(region)

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)
        depth, output_h, output_w = output_gradient.shape

        for d in range(depth):
            for i in range(output_h):
                for j in range(output_w):
                    region = self.input[
                        d,
                        i * self.stride: (i + 1) * self.stride,
                        j * self.stride: (j + 1) * self.stride
                    ]
                    
                    mask = (region == np.max(region))
                    input_gradient[
                        d,
                        i * self.stride: (i + 1) * self.stride,
                        j * self.stride: (j + 1) * self.stride
                    ] += output_gradient[d, i, j] * mask

        return input_gradient

