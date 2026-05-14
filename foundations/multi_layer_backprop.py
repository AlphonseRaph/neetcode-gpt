import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #

        # 0. Convert ALL inputs to NumPy arrays immediately so math and .T work
        x_np = np.array(x, ndmin=2)
        W1_np = np.array(W1)
        b1_np = np.array(b1)
        W2_np = np.array(W2)
        b2_np = np.array(b2)
        y_true_np = np.array(y_true, ndmin=2)

        # ------------------------------------------
        # FORWARD PROPAGATION
        # ------------------------------------------
        # Layer 1 + ReLU
        relu = (np.maximum(0, 
        np.dot(x_np, W1_np.T) + b1_np))                                   

        y_hat = ((np.dot(relu, # dot product of output of RELU              
                W2_np.T)                          
                + b2_np)) # addition of dot product of output of RELU + b2_np   

        # ------------------------------------------
        # LOSS CALCULATION
        # ------------------------------------------ 
        # The raw matrix for backprop
        error_diff = y_hat - y_true_np
        # The actual float value for final return dictionary (MSE)
        mse_loss = float(np.mean(error_diff ** 2))

        # ------------------------------------------
        # BACK PROPAGATION
        # ------------------------------------------
        # dw2 is derivative of mean squared error (loss function) with respect to W2 
        # (chain rule:
        # the derivative of y with respect to the derivative of W_2 OR:
        # The derivative of the Loss with respect to y_hat, 
        # multiplied by the derivative of y_hat with respect to W_2."
        # the derivative of y_hat with respect to W_2 is output of ReLU

        # derivative of Loss
        n = y_true_np.size  # gets the total number of elements in the array
        d_loss = (2 / n) * error_diff # The derivative

        dW2 = np.dot(d_loss.T, relu)
        # Calculate db2 by summing the errors across the batch
        """
        CALCULATING THE BIAS GRADIENT (db2) ACROSS A BATCH
        --------------------------------------------------
        1. The Batch Context: Our network processes multiple rows of data at once 
           (a "batch"). We dynamically know the size of this batch based on the 
           length of the input data (n). Because of this, our 'd_loss' matrix 
           contains a separate row of errors for every single example we processed.

        2. Why we sum (Shared Blame): We only have ONE set of biases (b2) for this 
           layer. Because that exact same bias was reused to make predictions for 
           every item in the batch, it shares the blame for all of those mistakes. 

        3. The Execution: To find the total gradient for this shared parameter, we 
           use np.sum(axis=0) to collapse the matrix vertically. This adds up the 
           errors from the entire batch into a single, unified adjustment vector.
        """
        db2 = np.sum(d_loss, axis=0, keepdims=True)

        """
        BACKPROPAGATION FOR W1 (THE CHAIN RULE)
        ---------------------------------------
        To calculate the gradient for the first layer's weights (dW1), the Chain Rule dictates 
        that we must calculate the derivative by multiplying the local gradients of each 
        operation, working backward from the final loss to the first layer.

        1. The Output Error (d_y_hat): We begin with the derivative of the Loss function 
        with respect to our final prediction. 
        2. Backward through Layer 2 (W2): We propagate this error backward through the second 
        linear layer by taking the dot product of the output error and the transposed W2. 
        This tells us how much the hidden layer contributed to the final mistake.
        3. Backward through Activation (ReLU): We filter this hidden error through the ReLU 
        derivative. If a neuron was inactive (input <= 0) during the forward pass, its 
        derivative is 0, so we zero out its error. If it was active, the error passes through.
        4. The Final W1 Gradient: Finally, we take the dot product of our original transposed 
        input data (x.T) and this filtered hidden error. This gives us the exact matrix 
        showing how a tiny change in W1 will impact the total loss.
        """
        # Backward through Layer 2: Propagate the output error backward to find the hidden layer's contribution.
        da1 = np.dot(d_loss, W2_np) 

        # Backward through Activation: Filter the hidden error through the ReLU derivative (zero out inactive neurons).
        dz1 = da1 * (relu > 0)

        # The Final W1 Gradient: Calculate the exact gradient matrix for W1 using the transposed input data.
        dW1 = np.dot(dz1.T, x_np)

        # Sum the filtered hidden errors across the batch dimension
        """
        CALCULATING db1: THE "HYDRAULIC PRESS" OF BLAME
        -----------------------------------------------
        The Situation: 
        Our network just processed a batch of 32 people. We generated 32 different 
        errors (dz1). However, we only have ONE set of biases (b1). That exact same 
        bias was added to every single person in the batch.
        
        The Solution (axis=0): 
        Because that one bias affected all 32 people, it must take the shared blame 
        for all 32 mistakes. We use np.sum(axis=0) like a vertical Hydraulic Press. 
        It takes the 32 rows of errors, crushes them straight down, and adds them 
        all together into a single, highly-concentrated row of blame.
        
        The Safety Net (keepdims=True):
        When NumPy crushes a matrix, it often strips away the brackets and turns it 
        into a flat, shape-less list (e.g., shape (10,) instead of (1, 10)). 
        keepdims=True forces NumPy to keep the outer brackets, ensuring our new 
        bias gradient perfectly matches the physical shape of our original bias!
        """
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # one quation : dW1 = np.dot(x_np.T, (np.dot(d_loss, W2_np.T) * (relu > 0)))

        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        # w.r.t means with respect to
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        result_dict = {
                'loss' : round(mse_loss, 4),
                'dW1' : np.round(dW1, 4).tolist(),
                'db1' : np.round(db1, 4).flatten().tolist(),
                'dW2' : np.round(dW2, 4).tolist(),
                'db2' : np.round(db2, 4).flatten().tolist()
        }

        return result_dict
