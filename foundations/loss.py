import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        # Force all predictions to be between 0.0000001 and 0.9999999
        y_pred_safe = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #
        term_1 = y_true * np.log(y_pred_safe)
        #
        term_2 = (1 - y_true) * np.log(1 - y_pred_safe)
        # Combine, make it positive, and return the average loss:
        # (Because logarithms of fractions (like $\log(0.9)$) always produce negative numbers)
        # (We take the mean because we usually calculate loss for a whole batch of data)
        loss = -np.mean(term_1 + term_2)

        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        # Force all predictions to be between 0.0000001 and 0.9999999
        y_pred_safe = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # First, we compress the grid horizontally. We add up the penalties,
        # to get a single loss score for that specific image.
        # We tell NumPy to add horizontally across the rows by using axis=1
        # Element-wise multiplication AND horizontal sum (Collapse the classes)
        # The negative sign is applied here to make the item losses positive
        item_losses = -np.sum(y_true * np.log(y_pred_safe), axis=1)
        # we take the average to get the final score for the whole batch.
        batch_loss = np.mean(item_losses)
        return round(batch_loss, 4)
