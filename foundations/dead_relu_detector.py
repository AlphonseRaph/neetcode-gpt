import torch
import torch.nn as nn
from typing import List


class Solution:
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions_list = []
        with torch.no_grad():
            current_x = x
            for layer in model.children():
                current_x = layer(current_x)
                if isinstance(layer, nn.ReLU ):
                    is_dead = (current_x <= 0.0).all(dim=0)
                    dead_fraction = is_dead.float().mean().item()
                    dead_fractions_list.append(dead_fraction)
        return dead_fractions_list

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise

        # # Check if the list is empty (this happens if the model had no ReLU layers)
        if not dead_fractions:
            return 'healthy'

        # 1. 'use_leaky_relu' if ANY layer has dead fraction > 0.5
        if max(dead_fractions) > 0.5:
            return 'use_leaky_relu'

        # 2. 'reinitialize' if the FIRST layer has dead fraction > 0.3
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. 'reduce_learning_rate' if strictly increases AND last layer > 0.1
        # Check if every element is strictly less than the one after it

        is_strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i+1] 
            for i in range(len(dead_fractions) - 1)
        )

        if len(dead_fractions) > 1 and is_strictly_increasing and dead_fractions[-1] > 0.1:
                return 'reduce_learning_rate'

        # 4 & 5. 'healthy' if max < 0.1, or as the default fallback
        # (Since both conditions return 'healthy', we can just return it)
        return 'healthy'
