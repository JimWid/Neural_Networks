import os
import numpy as np
import matplotlib.pyplot as plt
from numpy_nn.layers import Dense
from numpy_nn.losses import mse, mse_prime
from network_numpy import train, predict, save, load

# Data: Study hours vs Scores
hours  = np.array([2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,
                   7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,
                   2.7,4.8,3.8,6.9,7.8], dtype=float)

scores = np.array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
                   85, 62, 41, 42, 17, 95, 30, 24, 67, 69,
                   30, 54, 35, 76, 86], dtype=float)

# Normalization
hours_min, hours_max = hours.min(), hours.max()
scores_min, scores_max = scores.min(), scores.max()

X = (hours - hours_min) / (hours_max - hours_min)
y = (scores - scores_min) / (scores_max - scores_min)

# Reshaping for Dense Layer
X = X.reshape(-1, 1, 1)
y = y.reshape(-1, 1, 1)

# Linear Regression Model
network = [
    Dense(1, 1)
]

path = "models/regression_best.pkl"
epochs = 100
lr = 0.01

# Train if models doesn't exists
if os.path.exists(path):
    network = load(path)
else:
    train(network, mse, mse_prime, X, y, epochs, lr)
    save(network, path)

# Plotting Line
X_plot = np.linspace(0, 1, 100)
y_plot = [predict(network, np.array([[[x]]]))[0][0] for x in X_plot]

# Denormalizing for the plot axes
X_plot_real = X_plot * (hours_max - hours_min) + hours_min
y_plot_real = np.array(y_plot) * (scores_max - scores_min) + scores_min

plt.figure(figsize=(7, 4))
plt.scatter(hours, scores, color="steelblue", label="Real data", zorder=3)
plt.plot(X_plot_real, y_plot_real, color="red", linewidth=2, label="Model fit")
plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.legend()
plt.tight_layout()
plt.show()