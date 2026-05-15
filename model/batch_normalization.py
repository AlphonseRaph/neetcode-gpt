import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        # Convert inputs to numpy arrays for vectorized math
        X = np.array(x)
        G = np.array(gamma)
        B = np.array(beta)
        R_mean = np.array(running_mean)
        R_var = np.array(running_var)

        if training:
            # 1. Calculate batch stats per feature (axis=0)
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            # 2. Normalize using batch stats
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # 3. Apply affine transformation
            y = G * x_hat + B

            # 4. Update running statistics (Exponential Moving Average)
            R_mean = (1 - momentum) * R_mean + momentum * batch_mean
            R_var = (1 - momentum) * R_var + momentum * batch_var
            
        else:
            # Inference Mode: Use frozen running stats, ignore batch entirely
            x_hat = (x - R_mean) / np.sqrt(R_var + eps)

            # Apply affine transformation
            y = G * x_hat + B

        # Round outputs and convert back to Python lists to match type hints
        y_out = np.round(y, 4).tolist()
        r_mean_out = np.round(R_mean, 4).tolist()
        r_var_out = np.round(R_var, 4).tolist()
        return (y_out, r_mean_out, r_var_out)