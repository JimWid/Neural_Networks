import numpy as np
from numpy_nn.layers import Layer, Dense

# Tanh Function
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

# Base Activation Class
class Activation(Layer):
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime
		
	def forward(self, input):
		self.input = input
		return self.activation(self.input)
		
	def backward(self, output_gradient, learning_rate):
		return np.multiply(output_gradient, self.activation_prime(self.input))
    
# Activation Classes!	
class Tanh(Activation): # Hyperbolic Tangent
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation): # Sigmoid Function
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer): # Softmax Function
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient
        # n = np.size(self.output)
        #return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

class ReLU(Layer): # Rectified Linear Unit (ReLU)
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class BatchNormalization(Layer): # Batch Normalization!
    def __init__(self, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.scales = None
        self.bias = None

    def forward(self, input):
        self.input = input
        self.out_shape = input.shape

        self.mean = self.input.mean(axis=0)
        self.variance = 1. / np.sqrt(self.input.var(axis=0) + self.epsilon)

        self.input_norm = (self.input - self.mean) * self.variance
        self.output = self.input_norm.copy()

        if self.scales is not None:
            self.output *= self.scales

        if self.bias is not None:
            self.output +=self.bias
            
        return self.output
     
    def backward(self, output_gradient, learning_rate):
        input_shape = self.input.shape[0]
        inv_shape = 1. / input_shape

        bias_gradient = output_gradient.sum(axis=0)
        scales_gradient = (output_gradient * self.input_norm).sum(axis=0)

        if self.scales is not None:
            output_gradient *= self.scales

        mean_delta = (output_gradient * (-self.variance)).sum(axis=0)
        variance_delta = (output_gradient * (self.input - self.mean)).sum(axis=0) * -0.5 * self.variance**3

        delta = (output_gradient * self.variance
                + variance_delta * 2 * (self.input - self.mean) * inv_shape
                + mean_delta * inv_shape)

        if self.scales is not None:
            self.scales -= learning_rate * scales_gradient
        if self.bias is not None:
            self.bias -= learning_rate * bias_gradient

        return delta