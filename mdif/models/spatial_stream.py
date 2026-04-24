"""
Stream A: Spatial Stream

Backbone for the MDIF model, based on MobileNetV3-Small.
This stream processes the RGB image input and outputs a 576-dimensional feature vector.
"""

import torch
import torch.nn as nn
from torchvision import models


class SpatialStream(nn.Module):
    """
    Stream A: Spatial Stream (RGB Image Analysis).

    Backbone for the MDIF model, based on MobileNetV3-Small.
    This stream processes the RGB image input and outputs a 576-dimensional feature vector.

    The final classifier head is only used during standalone fine-tuning of the spatial stream.
    During fusion training, we will use the feature extractor part and freeze its weights.
    """

    def __init__(self, num_classes=3, pretrained=True):
        super(SpatialStream, self).__init__()
        self.backbone = models.mobilenet_v3_small(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        # Removing the final head to use it as a feature extractor
        self.feature_extractor = self.backbone.avgpool
        self.base_model = self.backbone.features

        self.classifier = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.base_model(x)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # The 576-dim spatial vector

        if return_features:
            return x

        return self.classifier(x)
