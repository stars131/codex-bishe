"""
Loss functions for network attack detection.
Includes Focal Loss, Label Smoothing, and other advanced loss functions.

Author: Network Attack Detection Project
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for each class. Can be:
            - None: no class weighting
            - float: weight for positive class (binary classification)
            - List[float]: weights for each class
        gamma: Focusing parameter (default: 2.0)
        reduction: 'none', 'mean', or 'sum'
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    def __init__(
        self,
        alpha: Optional[Union[float, List[float]]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)

        Returns:
            Focal loss value
        """
        num_classes = inputs.size(-1)

        # 计算交叉熵
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # 计算 p_t
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算 focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 应用 alpha 权重
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t

        # 计算 focal loss
        focal_loss = focal_weight * ce_loss

        # 应用 reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.

    Prevents the model from becoming overconfident.

    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)

        Returns:
            Label smoothing cross entropy loss
        """
        num_classes = inputs.size(-1)

        # 创建 soft targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # 计算 log softmax
        log_probs = F.log_softmax(inputs, dim=-1)

        # 计算 loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for handling extreme class imbalance.

    Paper: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2020)

    IMPORTANT: This loss function was originally designed for multi-label classification
    and uses sigmoid activation. For standard multi-class (mutually exclusive) classification,
    consider using FocalLoss instead, or set use_softmax=True.

    Args:
        gamma_neg: Focusing parameter for negative samples
        gamma_pos: Focusing parameter for positive samples
        clip: Clipping value for probability
        reduction: 'none', 'mean', or 'sum'
        use_softmax: If True, use softmax for multi-class classification (default: False)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean',
        use_softmax: bool = False
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.use_softmax = use_softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)

        Returns:
            Asymmetric loss value
        """
        num_classes = inputs.size(-1)

        # 转换为 one-hot
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # 计算概率 - 支持sigmoid(多标签)或softmax(多分类)
        if self.use_softmax:
            p = F.softmax(inputs, dim=-1)
        else:
            p = torch.sigmoid(inputs)

        # 正样本部分
        pos_part = targets_one_hot * torch.log(p.clamp(min=1e-8))
        pos_weight = (1 - p) ** self.gamma_pos

        # 负样本部分 (with probability shifting)
        p_neg = (p - self.clip).clamp(min=0)
        neg_part = (1 - targets_one_hot) * torch.log((1 - p_neg).clamp(min=1e-8))
        neg_weight = p_neg ** self.gamma_neg

        loss = -(pos_weight * pos_part + neg_weight * neg_part).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for handling class imbalance.

    Commonly used in segmentation but also effective for imbalanced classification.

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)

        Returns:
            Dice loss value
        """
        num_classes = inputs.size(-1)

        # 转换为概率
        probs = F.softmax(inputs, dim=-1)

        # 转换为 one-hot
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # 计算 Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines multiple losses.

    Args:
        losses: List of (loss_module, weight) tuples
    """

    def __init__(self, losses: List[tuple]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)

        Returns:
            Combined loss value
        """
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(inputs, targets)
        return total_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    Paper: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

    Args:
        samples_per_class: Number of samples for each class
        beta: Hyperparameter for effective number calculation (default: 0.9999)
        loss_type: Base loss type ('ce', 'focal', 'sigmoid')
        gamma: Gamma for focal loss
    """

    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = 'focal',
        gamma: float = 2.0
    ):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma

        # 计算有效样本数和权重
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)

        Returns:
            Class-balanced loss value
        """
        weights = self.weights.to(inputs.device)

        if self.loss_type == 'ce':
            return F.cross_entropy(inputs, targets, weight=weights)
        elif self.loss_type == 'focal':
            focal = FocalLoss(gamma=self.gamma, reduction='none')
            loss = focal(inputs, targets)
            # 应用类别权重
            weight_t = weights.gather(0, targets)
            return (weight_t * loss).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for learning better representations.

    Paper: "Supervised Contrastive Learning" (Khosla et al., 2020)

    Args:
        temperature: Temperature scaling parameter
        base_temperature: Base temperature
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: Hidden representations, shape (N, D)
            labels: Ground truth labels, shape (N,)
            mask: Contrastive mask (optional)

        Returns:
            Contrastive loss value
        """
        device = features.device
        batch_size = features.size(0)

        # 归一化特征
        features = F.normalize(features, dim=1)

        # 创建标签 mask
        labels = labels.view(-1, 1)
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)

        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 移除对角线（自身）
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # 计算 log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # 计算平均 log-likelihood
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class CenterLoss(nn.Module):
    """
    Center Loss for learning discriminative features.

    Paper: "A Discriminative Feature Learning Approach for Deep Face Recognition" (Wen et al., 2016)

    Args:
        num_classes: Number of classes
        feature_dim: Feature dimension
        alpha: Learning rate for center update
    """

    def __init__(self, num_classes: int, feature_dim: int, alpha: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha

        # 初始化类中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Hidden representations, shape (N, D)
            labels: Ground truth labels, shape (N,)

        Returns:
            Center loss value
        """
        batch_size = features.size(0)

        # 获取对应类别的中心
        centers_batch = self.centers[labels]

        # 计算距离
        loss = F.mse_loss(features, centers_batch)

        return loss

    def update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """更新类中心（在训练循环中调用）"""
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = labels == i
                if mask.sum() > 0:
                    class_features = features[mask]
                    diff = class_features.mean(dim=0) - self.centers[i]
                    self.centers[i] += self.alpha * diff


# ============================================
# 损失函数工厂
# ============================================

def create_loss_function(
    loss_type: str,
    num_classes: int = 2,
    class_weights: Optional[List[float]] = None,
    samples_per_class: Optional[List[int]] = None,
    **kwargs
) -> nn.Module:
    """
    损失函数工厂

    支持的主损失函数类型:
        - cross_entropy: 标准交叉熵损失
        - focal: Focal Loss（处理类别不平衡）
        - label_smoothing: 标签平滑交叉熵
        - asymmetric: 非对称损失
        - dice: Dice Loss
        - class_balanced: 类别平衡损失（需提供 samples_per_class）
        - combined: 组合损失

    注意: ContrastiveLoss 和 CenterLoss 为辅助损失函数，
    接收特征向量而非分类 logits，需单独实例化并与主损失函数配合使用。

    Args:
        loss_type: 损失函数类型
        num_classes: 类别数
        class_weights: 类别权重
        samples_per_class: 每类样本数（用于class balanced loss）
        **kwargs: 额外参数

    Returns:
        损失函数模块
    """
    if loss_type == 'cross_entropy':
        weight = torch.tensor(class_weights) if class_weights else None
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )

    elif loss_type == 'focal':
        return FocalLoss(
            alpha=class_weights,
            gamma=kwargs.get('gamma', 2.0),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )

    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            smoothing=kwargs.get('smoothing', 0.1)
        )

    elif loss_type == 'asymmetric':
        return AsymmetricLoss(
            gamma_neg=kwargs.get('gamma_neg', 4.0),
            gamma_pos=kwargs.get('gamma_pos', 1.0),
            use_softmax=kwargs.get('use_softmax', False)
        )

    elif loss_type == 'dice':
        return DiceLoss()

    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class_balanced loss")
        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            beta=kwargs.get('beta', 0.9999),
            gamma=kwargs.get('gamma', 2.0)
        )

    elif loss_type == 'combined':
        # 默认组合: Focal + Dice
        return CombinedLoss([
            (FocalLoss(gamma=2.0), 0.7),
            (DiceLoss(), 0.3)
        ])

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
