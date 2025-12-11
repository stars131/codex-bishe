"""
Deep learning models for network attack detection.
Supports CNN, LSTM, Transformer, and Fusion architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionFusion(nn.Module):
    """Attention-based multi-source fusion module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention weights to fuse features."""
        attn_weights = F.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)


class FusionNet(nn.Module):
    """
    Multi-source data fusion network for attack detection.
    Combines traffic and log features using attention mechanism.
    """

    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Traffic feature encoder
        self.traffic_encoder = nn.Sequential(
            nn.Linear(traffic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Log feature encoder
        self.log_encoder = nn.Sequential(
            nn.Linear(log_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Attention-based fusion
        self.fusion = AttentionFusion(hidden_dim, hidden_dim // 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        traffic_features: torch.Tensor,
        log_features: torch.Tensor
    ) -> torch.Tensor:
        # Encode each source
        traffic_encoded = self.traffic_encoder(traffic_features)
        log_encoded = self.log_encoder(log_features)

        # Stack for attention
        combined = torch.stack([traffic_encoded, log_encoded], dim=1)

        # Fuse with attention
        fused = self.fusion(combined)

        # Classify
        return self.classifier(fused)


class SingleSourceNet(nn.Module):
    """Single source network for baseline comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
