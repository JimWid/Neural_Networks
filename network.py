import numpy as np
import pickle

# Prediction Helper Function
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# Training Function
def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):

            # Forward Pass
            output = predict(network, x)

            # Error
            error += loss(y, output)

            # Backward Pass (Updgrading Weights)
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)

        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

# Testing Function
def test(network, x_test, y_test, n=10):
    correct = 0

    for i in range(n):
        output = x_test[i]
        for layer in network:
            output = layer.forward(output)

        if np.argmax(output) == np.argmax(y_test[i]):
            correct += 1

        print(f"Prediction: {np.argmax(output)}, Actual: {np.argmax(y_test[i])}")
    
    print(f"Accuracy: {correct / n}")

# Save Model Function
def save(network, path):
    with open(path, "wb") as f:
        pickle.dump(network, f)
    print(f"Network saved in 'models' folder.")

# Load Model Function
def load(path):
    with open(path, "rb") as f:
        network = pickle.load(f)
    print(f"Model from {path} loaded succesfully.")

    return network