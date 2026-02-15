"""
数据集和数据加载器单元测试
"""
import pytest
import torch
import numpy as np
from src.data.dataset import MultiSourceDataset, NetworkAttackDataset, create_multi_source_loaders


class TestMultiSourceDataset:
    """测试多源数据集"""

    def test_getitem(self):
        """测试 __getitem__ 返回正确的三元组"""
        np.random.seed(42)
        n = 100
        s1 = np.random.randn(n, 20).astype(np.float32)
        s2 = np.random.randn(n, 15).astype(np.float32)
        y = np.random.randint(0, 5, n)

        dataset = MultiSourceDataset(s1, s2, y)

        assert len(dataset) == n

        source1, source2, label = dataset[0]
        assert isinstance(source1, torch.Tensor)
        assert isinstance(source2, torch.Tensor)
        assert source1.shape == (20,)
        assert source2.shape == (15,)

    def test_dataset_dtypes(self):
        """测试数据类型正确"""
        s1 = np.random.randn(50, 10).astype(np.float32)
        s2 = np.random.randn(50, 8).astype(np.float32)
        y = np.random.randint(0, 3, 50)

        dataset = MultiSourceDataset(s1, s2, y)
        source1, source2, label = dataset[0]

        assert source1.dtype == torch.float32
        assert source2.dtype == torch.float32
        assert label.dtype == torch.int64


class TestNetworkAttackDataset:
    """测试单源数据集"""

    def test_getitem(self):
        """测试 __getitem__"""
        X = np.random.randn(50, 30).astype(np.float32)
        y = np.random.randint(0, 5, 50)

        dataset = NetworkAttackDataset(X, y)
        features, label = dataset[0]

        assert features.shape == (30,)
        assert isinstance(label, torch.Tensor)


class TestMultiSourceLoaders:
    """测试多源数据加载器"""

    def test_create_loaders(self):
        """测试 create_multi_source_loaders 返回正确结构"""
        np.random.seed(42)
        n_train, n_val, n_test = 200, 50, 50

        data_dict = {
            'X1_train': np.random.randn(n_train, 20).astype(np.float32),
            'X1_val': np.random.randn(n_val, 20).astype(np.float32),
            'X1_test': np.random.randn(n_test, 20).astype(np.float32),
            'X2_train': np.random.randn(n_train, 15).astype(np.float32),
            'X2_val': np.random.randn(n_val, 15).astype(np.float32),
            'X2_test': np.random.randn(n_test, 15).astype(np.float32),
            'y_train': np.random.randint(0, 5, n_train),
            'y_val': np.random.randint(0, 5, n_val),
            'y_test': np.random.randint(0, 5, n_test),
        }

        loaders = create_multi_source_loaders(data_dict, batch_size=32, num_workers=0)

        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders

        # 测试能正常迭代
        batch = next(iter(loaders['train']))
        assert len(batch) == 3  # source1, source2, labels
        s1, s2, labels = batch
        assert s1.shape[1] == 20
        assert s2.shape[1] == 15
