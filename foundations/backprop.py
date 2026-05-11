import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # -----------------------------
        # FORWARD PASS
        # -----------------------------
        # Linear combination
        z = (np.dot(x, w) + b )
        # Sigmoid activation
        y_hat = 1 /( 1 + np.exp(-z))
        # Actual Mean Squared Error loss
        mse_loss = 0.5 * (y_hat - y_true) ** 2

        # -----------------------------
        # BACKWARD PASS
        # -----------------------------

        # dL/dy_hat
        dL_dyhat = y_hat - y_true

        #  dy_hat/dz  (derivative of sigmoid)
        dyhat_dz = y_hat * (1 - y_hat)
        # dL/dz  (chain rule) / delta
        dL_dz = dyhat_dz * dL_dyhat
        # blaming specific weights 
        # dL/dw
        dL_dw = dL_dz * x
        # blaming bias (dL/db)
        dL_db = dL_dz 


        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        return np.round(dL_dw, 5), round(float(dL_db), 5)
