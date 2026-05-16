import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        # Extract the number of features (the second dimension of X)
        n_features = X.shape[1]
        # n is the number of samples (rows in X)
        n = X.shape[0]

        w = np.zeros(n_features, dtype=np.float64)
        b = np.float64(0.0)
        for i in  range(epochs):

            y_hat = np.dot(X, w) + b
            error = y_hat - y
            mse = (1/n) * sum((error) ** 2)

            # Derivative with respect to weights (dw)
            dw = (2 / n) * np.dot(X.T, error)

            # Derivative with respect to bias (db)
            db = (2 / n) * np.sum(error)

            w = w - (lr * dw)
            b = b - (lr * db)

        return (np.round(w, 5), round(b, 5))
