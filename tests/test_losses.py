"""
损失函数单元测试

测试所有损失函数类型的前向传播和梯度流
"""
import pytest
import torch
from src.models.losses import (
    create_loss_function, FocalLoss, LabelSmoothingCrossEntropy,
    DiceLoss, AsymmetricLoss, ClassBalancedLoss, CombinedLoss
)


@pytest.fixture
def logits_and_labels():
    """生成测试用的 logits 和 labels"""
    torch.manual_seed(42)
    batch_size = 16
    num_classes = 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    return logits, labels, num_classes


class TestLossFactory:
    """测试损失函数工厂"""

    @pytest.mark.parametrize("loss_type", [
        "cross_entropy", "focal", "label_smoothing", "asymmetric", "dice", "combined"
    ])
    def test_factory_creates_loss(self, loss_type, logits_and_labels):
        """测试工厂函数能正确创建各类型损失"""
        logits, labels, num_classes = logits_and_labels
        loss_fn = create_loss_function(loss_type, num_classes=num_classes)
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0, f"{loss_type} 损失应为标量"
        assert not torch.isnan(loss), f"{loss_type} 损失为 NaN"
        assert loss.item() >= 0, f"{loss_type} 损失应为非负值"

    def test_class_balanced_requires_samples(self):
        """测试 class_balanced 必须提供 samples_per_class"""
        with pytest.raises(ValueError, match="samples_per_class required"):
            create_loss_function('class_balanced', num_classes=5)

    def test_class_balanced_with_samples(self, logits_and_labels):
        """测试 class_balanced 正常创建"""
        logits, labels, num_classes = logits_and_labels
        loss_fn = create_loss_function(
            'class_balanced',
            num_classes=num_classes,
            samples_per_class=[100, 200, 150, 80, 50]
        )
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)

    def test_invalid_loss_type(self):
        """测试无效损失类型"""
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss_function('invalid_loss')


class TestFocalLoss:
    """测试 Focal Loss"""

    def test_gradient_flow(self, logits_and_labels):
        """测试梯度是否正常传播"""
        logits, labels, _ = logits_and_labels
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_gamma_effect(self, logits_and_labels):
        """测试 gamma 参数的效果：gamma 越大，对易分类样本的惩罚越大"""
        logits, labels, _ = logits_and_labels
        loss_g0 = FocalLoss(gamma=0.0)(logits, labels)
        loss_g2 = FocalLoss(gamma=2.0)(logits, labels)
        # gamma=0 等效于标准 CE，gamma>0 损失应更小
        assert loss_g2 <= loss_g0


class TestLabelSmoothingLoss:
    """测试标签平滑损失"""

    def test_smoothing_zero(self, logits_and_labels):
        """smoothing=0 应接近标准 CE"""
        logits, labels, _ = logits_and_labels
        ce_loss = torch.nn.CrossEntropyLoss()(logits, labels)
        ls_loss = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, labels)
        assert abs(ce_loss.item() - ls_loss.item()) < 0.01
