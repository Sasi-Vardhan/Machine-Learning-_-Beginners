import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(-3, 3, 100)
Y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Convert to column vectors
X_train = np.c_[np.ones(X.shape), X]  # Add bias term
Y_train = Y.reshape(-1, 1)

def gaussian_weight(x, x_query, tau):
    return np.exp(-(x - x_query) ** 2 / (2 * tau ** 2))

def locally_weighted_regression(x_train, y_train, x_query, tau=0.5):
    W = np.diag(gaussian_weight(x_train[:, 1], x_query[1], tau))
    theta = np.linalg.pinv(x_train.T @ W @ x_train) @ x_train.T @ W @ y_train
    return x_query @ theta

# Query points
X_query = np.linspace(-3, 3, 100)
X_query_bias = np.c_[np.ones(X_query.shape), X_query]

# Predict using LWR
tau = 0.3  # Bandwidth
Y_pred = np.array([locally_weighted_regression(X_train, Y_train, x, tau) for x in X_query_bias])

# Plot results
plt.scatter(X, Y, label="Training Data")
plt.plot(X_query, Y_pred, color='red', label="LWR Predictions")
plt.legend()
plt.show()
