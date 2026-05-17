import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        
        # Turn off gradient tracking for the entire function
        with torch.no_grad(): 
            # This variable will hold our data as it flows through the network
            current_x = x
            # Iterate through the model layer by layer
            for layer in model.children():
                # Pass the data through the current layer
                current_x = layer(current_x)
                # Check if the layer we just passed through is an nn.Linear layer
                if isinstance(layer, nn.Linear):
                    # Calculate the raw stats
                    mean = current_x.mean().item()
                    std = current_x.std().item()
                    # Dead fraction: Create a boolean tensor of where values == 0, 
                    # Check if <= 0 across ALL items in the batch (dim=0)
                    is_dead = (current_x <= 0.0).all(dim=0)
                    # convert to float (1.0 for True, 0.0 for False), and take the mean
                    dead_fraction = is_dead.float().mean().item()
                    # --- WHY WE USE .item() ---
                    # 1. Extracts the raw Python float/int from a 1-element tensor.
                    # 2. Fixes compatibility (allows standard round(), saving to JSON/CSV).
                    # 3. PREVENTS MEMORY LEAKS: Detaches the value from PyTorch's autograd graph.
                    #    (Appending raw tensors while training to lists stores the entire math history in RAM!):
                    # in this cause the entire history won't be saved because of torch.no_grad but storing a tensor,
                    # will still occupy a larger memory

                    # 4. Round to 4 decimals and append to our list
                    stats.append(
                        {'mean': round(mean, 4), 
                        'std': round(std, 4),
                        'dead_fraction': round(dead_fraction, 4)
                        }
                        )
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with C
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []

        # 1. Forward + backward pass
        model.zero_grad()
        predictions = model(x)

        # Instantiate the loss class, then calculate the loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y)
        loss.backward()

        # 2. Inspect the layers  
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                # Safety check: ensure the gradient actually exists
                if grad is not None:
                    # Added () to mean and std, and .item() to norm
                    mean = grad.mean().item()
                    std = grad.std().item()
                    norm = torch.norm(grad, p=2).item()

                    stats.append(
                        {'mean': round(mean, 4), 
                        'std': round(std, 4),
                        'norm' : round(norm, 4)
                        }
                    )

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        # 1. Dead neurons: if ANY layer has dead_fraction > 0.5
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        # 2. Exploding gradients: if ANY layer gradient norm > 1000
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'
        
        # 3. Vanishing gradients: if LAST layer gradient norm < 1e-5
        # We use index [-1] to grab only the last layer's stats from the list
        if len(gradient_stats) > 0 and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        # 4. Activation std for ALL layers
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10.0:
                return 'exploding_gradients'

        # 5. If it passes all checks
        return 'healthy'
