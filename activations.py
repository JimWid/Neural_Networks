import numpy as np
from layers import Layer

class Activation(Layer):
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime
		
	def forward(self, input):
		self.input = input
		return self.activation(self.input)
		
	def backward(self, output_gradient, learning_rate):
		return np.multiply(output_gradient, self.activation_prime(self.input))

# Activation functions!	
class Tanh(Activation): # Hyperbolic Tangent
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation): # Sigmoid Function
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

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