import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        new_z = z - np.max(z)

        exp_new_z = np.exp(new_z)

        sum_exp_new_z = np.sum(exp_new_z )

        probabilities = exp_new_z  / sum_exp_new_z

        return np.round(probabilities, 4)
        
