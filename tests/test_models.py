"""
模型单元测试

测试 FusionNet、SingleSourceNet、EnsembleFusionNet 的前向传播
覆盖所有编码器类型和融合方法
"""
import pytest
import torch
from src.models.fusion_net import create_model, FusionNet, SingleSourceNet


class TestFusionNetEncoders:
    """测试不同编码器类型"""

    @pytest.mark.parametrize("encoder_type", ["mlp", "cnn", "lstm", "transformer"])
    def test_encoder_forward(self, sample_data, sample_dims, encoder_type):
        """测试各编码器类型的前向传播"""
        model = create_model(
            model_type='fusion_net',
            traffic_dim=sample_dims['source1_dim'],
            log_dim=sample_dims['source2_dim'],
            num_classes=sample_dims['num_classes'],
            config={
                'hidden_dim': sample_dims['hidden_dim'],
                'dropout': 0.1,
                'encoder_type': encoder_type,
                'fusion_type': 'attention',
                'num_layers': 2,
                'num_heads': 4
            }
        )
        model.eval()

        with torch.no_grad():
            logits, attention = model(sample_data['source1'], sample_data['source2'])

        assert logits.shape == (sample_dims['batch_size'], sample_dims['num_classes'])
        assert attention.shape[0] == sample_dims['batch_size']


class TestFusionMethods:
    """测试不同融合方法"""

    @pytest.mark.parametrize("fusion_type", [
        "attention", "multi_head", "cross", "gated", "bilinear", "concat"
    ])
    def test_fusion_forward(self, sample_data, sample_dims, fusion_type):
        """测试各融合方法的前向传播"""
        model = create_model(
            model_type='fusion_net',
            traffic_dim=sample_dims['source1_dim'],
            log_dim=sample_dims['source2_dim'],
            num_classes=sample_dims['num_classes'],
            config={
                'hidden_dim': sample_dims['hidden_dim'],
                'dropout': 0.1,
                'encoder_type': 'mlp',
                'fusion_type': fusion_type,
                'num_layers': 2,
                'num_heads': 4
            }
        )
        model.eval()

        with torch.no_grad():
            logits, attention = model(sample_data['source1'], sample_data['source2'])

        assert logits.shape == (sample_dims['batch_size'], sample_dims['num_classes'])
        assert not torch.isnan(logits).any(), f"融合方法 {fusion_type} 输出包含 NaN"


class TestSingleSourceNet:
    """测试单源基线模型"""

    def test_forward(self, sample_data, sample_dims):
        """测试 SingleSourceNet 前向传播"""
        total_dim = sample_dims['source1_dim'] + sample_dims['source2_dim']
        model = create_model(
            model_type='single_source',
            traffic_dim=sample_dims['source1_dim'],
            log_dim=sample_dims['source2_dim'],
            num_classes=sample_dims['num_classes'],
            config={
                'hidden_dim': sample_dims['hidden_dim'],
                'dropout': 0.1,
                'encoder_type': 'mlp',
                'num_layers': 2
            }
        )
        model.eval()

        combined = torch.cat([sample_data['source1'], sample_data['source2']], dim=1)
        with torch.no_grad():
            logits = model(combined)

        assert logits.shape == (sample_dims['batch_size'], sample_dims['num_classes'])


class TestEnsembleFusionNet:
    """测试集成融合模型"""

    def test_forward(self, sample_data, sample_dims):
        """测试 EnsembleFusionNet 前向传播"""
        model = create_model(
            model_type='ensemble',
            traffic_dim=sample_dims['source1_dim'],
            log_dim=sample_dims['source2_dim'],
            num_classes=sample_dims['num_classes'],
            config={
                'hidden_dim': sample_dims['hidden_dim'],
                'dropout': 0.1,
                'fusion_types': ['attention', 'gated']
            }
        )
        model.eval()

        with torch.no_grad():
            logits, attention = model(sample_data['source1'], sample_data['source2'])

        assert logits.shape == (sample_dims['batch_size'], sample_dims['num_classes'])


class TestModelFactory:
    """测试模型工厂函数"""

    def test_invalid_model_type(self, sample_dims):
        """测试无效模型类型"""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(
                model_type='invalid_type',
                traffic_dim=sample_dims['source1_dim'],
                log_dim=sample_dims['source2_dim'],
                num_classes=sample_dims['num_classes']
            )

    def test_gradient_flow(self, sample_data, sample_dims):
        """测试梯度流是否正常"""
        model = create_model(
            model_type='fusion_net',
            traffic_dim=sample_dims['source1_dim'],
            log_dim=sample_dims['source2_dim'],
            num_classes=sample_dims['num_classes'],
            config={'hidden_dim': sample_dims['hidden_dim'], 'encoder_type': 'mlp', 'fusion_type': 'attention'}
        )
        model.train()

        logits, _ = model(sample_data['source1'], sample_data['source2'])
        loss = torch.nn.CrossEntropyLoss()(logits, sample_data['labels'])
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "梯度未正确传播"
