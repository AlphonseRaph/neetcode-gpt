import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        # YHAT = X⋅W (dot product of feature matrix and weight vector)
        return np.round(np.dot(X, weights), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        # 1. Calculate the squared errors for the entire array
        squared_errors = (model_prediction - ground_truth) ** 2
        # 2. Find the mean (which automatically sums and divides by length)
        mse = np.mean(squared_errors)


        return round(float(mse), 5)
