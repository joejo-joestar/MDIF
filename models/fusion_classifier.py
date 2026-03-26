import torch.nn as nn


class MDIFFusionClassifier(nn.Module):
    def __init__(self, input_dim=777, hidden_dim=256, num_classes=3):
        super(MDIFFusionClassifier, self).__init__()
        # Architecture defined in Section 6.2/8.0 of the proposal
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
