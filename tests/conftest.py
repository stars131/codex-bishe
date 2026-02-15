"""
测试共享 fixtures
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_dims():
    """标准测试维度"""
    return {
        'source1_dim': 22,
        'source2_dim': 19,
        'num_classes': 5,
        'hidden_dim': 64,
        'batch_size': 16,
    }


@pytest.fixture
def sample_data(sample_dims):
    """生成测试用的样本数据"""
    torch.manual_seed(42)
    np.random.seed(42)

    bs = sample_dims['batch_size']
    s1_dim = sample_dims['source1_dim']
    s2_dim = sample_dims['source2_dim']
    n_cls = sample_dims['num_classes']

    return {
        'source1': torch.randn(bs, s1_dim),
        'source2': torch.randn(bs, s2_dim),
        'labels': torch.randint(0, n_cls, (bs,)),
    }


@pytest.fixture
def device():
    """计算设备"""
    return torch.device('cpu')


@pytest.fixture
def class_names():
    """测试用类别名称"""
    return ['normal', 'dos', 'probe', 'r2l', 'u2r']
