from numpy_nn.layers import Layer
from numpy_nn.convolutional import Convolution
from numpy_nn.activations import ReLU, BatchNormalization

class ResidualBlock(Layer):
    def __init__(self, input_shape, kernel_size, depth, epsilon=1e-8, padding=1):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.epsilon = epsilon
        self.padding = padding

        # Conv1 output shape
        conv1_h = input_height + 2 * padding - kernel_size + 1
        conv1_w = input_width + 2 * padding - kernel_size + 1
        conv1_shape = (depth, conv1_h, conv1_w)

        self.conv1 = Convolution(input_shape, kernel_size, depth, padding)
        self.norm1 = BatchNormalization(epsilon)
        self.relu1 = ReLU()
        self.conv2 = Convolution(conv1_shape, kernel_size, input_depth, padding)
        self.norm2 = BatchNormalization(epsilon)
        self.relu2 = ReLU()

    def forward(self, input):
        self.input = input

        x = self.conv1.forward(input)
        x = self.norm1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.norm2.forward(x)

        x = x + self.input # Residuals!!!

        x = self.relu2.forward(x)

        return x
    
    def backward(self, output_gradient, learning_rate):
        
        grad = self.relu2.backward(output_gradient, learning_rate)
        residual_grad = grad

        grad = self.norm2.backward(grad, learning_rate)
        grad = self.conv2.backward(grad, learning_rate)
        grad = self.relu1.backward(grad, learning_rate)
        grad = self.norm1.backward(grad, learning_rate)
        grad = self.conv1.backward(grad, learning_rate)

        return grad + residual_grad