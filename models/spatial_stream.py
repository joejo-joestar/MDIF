import torch
import torch.nn as nn
from torchvision import models


class SpatialStream(nn.Module):
    def __init__(self, num_classes=3):
        super(SpatialStream, self).__init__()
        # Load pre-trained MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

        # We need the 576-dim vector from the layer before the final classifier
        # Removing the final head to use it as a feature extractor
        self.feature_extractor = self.backbone.avgpool
        self.base_model = self.backbone.features

        # Internal classifier for standalone fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.base_model(x)
        x = self.feature_extractor(x)
        # This is the 576-dim spatial vector
        x = torch.flatten(x, 1)

        if return_features:
            return x

        return self.classifier(x)
