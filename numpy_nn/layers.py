import numpy as np

# Layer Base Class
class Layer:
	def __init__(self):
		self.input = None
		self.output = None
		
	def forward(self, input):
		pass
		
	def backward(self, output_gradient, learning_rate):
		pass

# Different layer types!
class Dense(Layer):
	def __init__(self, input_size, output_size):
		self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
		self.bias = np.zeros((output_size, 1))
		
	def forward(self, input):
		self.input = input
		return np.dot(self.weights, self.input) + self.bias 
		
	def backward(self, output_gradient, learning_rate):
		weights_gradient = np.dot(output_gradient, self.input.T)
		self.weights -= learning_rate * weights_gradient
		self.bias -= learning_rate * output_gradient
		return np.dot(self.weights.T, output_gradient)
	