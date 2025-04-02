import torch
import torch.nn.functional as F
import torch.nn as nn

class EntropyCalculator(nn.Module):
    def __init__(self):
        super(EntropyCalculator, self).__init__()

    def forward(self, feature_map):
        # Flatten the feature map
        flattened_map = feature_map.view(-1)

        # Calculate the probability distribution
        unique_elements, counts = torch.unique(flattened_map, return_counts=True)
        probabilities = counts.float() / len(flattened_map)

        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-12))  # Add a small epsilon to avoid log(0)

        return entropy
