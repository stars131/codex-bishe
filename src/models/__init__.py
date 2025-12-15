"""
Models module for network attack detection.
"""
from src.models.fusion_net import (
    FusionNet,
    SingleSourceNet,
    EnsembleFusionNet,
    create_model,
    # 融合模块
    AttentionFusion,
    MultiHeadAttentionFusion,
    CrossAttentionFusion,
    GatedFusion,
    BilinearFusion,
    # 编码器
    MLPEncoder,
    CNNEncoder,
    LSTMEncoder,
    TransformerEncoder,
    # 基础模块
    ResidualBlock,
    PositionalEncoding
)

from src.models.losses import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    AsymmetricLoss,
    DiceLoss,
    CombinedLoss,
    ClassBalancedLoss,
    ContrastiveLoss,
    CenterLoss,
    create_loss_function
)

__all__ = [
    # 主要模型
    'FusionNet',
    'SingleSourceNet',
    'EnsembleFusionNet',
    'create_model',
    # 融合模块
    'AttentionFusion',
    'MultiHeadAttentionFusion',
    'CrossAttentionFusion',
    'GatedFusion',
    'BilinearFusion',
    # 编码器
    'MLPEncoder',
    'CNNEncoder',
    'LSTMEncoder',
    'TransformerEncoder',
    # 基础模块
    'ResidualBlock',
    'PositionalEncoding',
    # 损失函数
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'AsymmetricLoss',
    'DiceLoss',
    'CombinedLoss',
    'ClassBalancedLoss',
    'ContrastiveLoss',
    'CenterLoss',
    'create_loss_function'
]
