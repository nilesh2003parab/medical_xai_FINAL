"""
FusionModel: CNN (ResNet18) backbone + optional metadata fusion.
Designed for binary classification: 0 = Normal, 1 = Pneumonia.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FusionModel(nn.Module):
    """
    A fusion model combining a ResNet18 CNN backbone with optional
    tabular metadata input for enhanced medical image classification.
    
    Architecture:
        - CNN branch: ResNet18 pretrained on ImageNet
        - Fusion head: FC layers combining CNN features + metadata (optional)
        - Output: 2-class softmax (Normal / Pneumonia)
    """

    def __init__(self, num_classes: int = 2, metadata_dim: int = 0, dropout: float = 0.4):
        super(FusionModel, self).__init__()
        self.metadata_dim = metadata_dim

        # ── CNN Backbone ──────────────────────────────────────────────────
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        cnn_out_features = self.cnn.fc.in_features  # 512

        # Replace the final FC with Identity so we can extract features
        self.cnn.fc = nn.Identity()

        # ── Fusion Head ──────────────────────────────────────────────────
        fusion_in = cnn_out_features + metadata_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:        Image tensor  [B, 3, H, W]
            metadata: Optional tabular features [B, metadata_dim]
        Returns:
            logits:   [B, num_classes]
        """
        features = self.cnn(x)  # [B, 512]

        if self.metadata_dim > 0 and metadata is not None:
            features = torch.cat([features, metadata], dim=1)

        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN feature vector without classification head."""
        with torch.no_grad():
            return self.cnn(x)
