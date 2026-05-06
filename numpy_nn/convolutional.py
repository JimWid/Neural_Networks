import numpy as np
from numpy_nn.layers import Layer
from scipy import signal

class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth, padding=0):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.padding = padding

        self.input_shape = input_shape
        self.input_depth = input_depth

        output_height = input_height + 2 * padding - kernel_size + 1
        output_width = input_width + 2 * padding - kernel_size + 1
        self.output_shape = (depth, output_height, output_width)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        self.fan_in = input_depth * kernel_size * kernel_size
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2 / self.fan_in)
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        # Padding
        if self.padding > 0:
            self.padded_input = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            self.padded_input = input

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.padded_input[j], self.kernels[i, j], mode='valid')
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.padded_input[j], output_gradient[i], mode='valid')

                full_grad = signal.convolve2d(output_gradient[i], self.kernels[i, j], mode='full')
                if self.padding > 0:
                    full_grad = full_grad[self.padding:-self.padding, self.padding:-self.padding]
                
                input_gradient[j] += full_grad

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient