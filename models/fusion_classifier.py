"""
Fusion Classifier for the MDIF model.
This module takes the concatenated feature vector from both streams (spatial and frequency) and outputs the final classification result (real vs. fake vs inpainted).

The architecture is a simple feedforward neural network with two hidden layers,
designed to learn the complex relationships between the spatial and frequency features for accurate classification.
"""

import torch.nn as nn


class MDIFFusionClassifier(nn.Module):
    def __init__(self, input_dim=777, hidden_dim=256, num_classes=3):
        super(MDIFFusionClassifier, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fusion_layer(x)
