"""
Dataset and dataloader tests.
"""
import numpy as np
import torch

from src.data.dataset import MultiSourceDataset, NetworkAttackDataset, create_multi_source_loaders


class TestMultiSourceDataset:
    def test_getitem(self):
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
        assert label.dtype == torch.int64

    def test_dataset_dtypes(self):
        s1 = np.random.randn(50, 10).astype(np.float32)
        s2 = np.random.randn(50, 8).astype(np.float32)
        y = np.random.randint(0, 3, 50)

        dataset = MultiSourceDataset(s1, s2, y)
        source1, source2, label = dataset[0]
        assert source1.dtype == torch.float32
        assert source2.dtype == torch.float32
        assert label.dtype == torch.int64


class TestNetworkAttackDataset:
    def test_getitem(self):
        X = np.random.randn(50, 30).astype(np.float32)
        y = np.random.randint(0, 5, 50)

        dataset = NetworkAttackDataset(X, y)
        features, label = dataset[0]
        assert features.shape == (30,)
        assert isinstance(label, torch.Tensor)


class TestMultiSourceLoaders:
    def test_create_loaders(self):
        np.random.seed(42)
        n_train, n_val, n_test = 200, 50, 50

        data_dict = {
            "X1_train": np.random.randn(n_train, 20).astype(np.float32),
            "X1_val": np.random.randn(n_val, 20).astype(np.float32),
            "X1_test": np.random.randn(n_test, 20).astype(np.float32),
            "X2_train": np.random.randn(n_train, 15).astype(np.float32),
            "X2_val": np.random.randn(n_val, 15).astype(np.float32),
            "X2_test": np.random.randn(n_test, 15).astype(np.float32),
            "y_train": np.random.randint(0, 5, n_train),
            "y_val": np.random.randint(0, 5, n_val),
            "y_test": np.random.randint(0, 5, n_test),
        }

        loaders = create_multi_source_loaders(data_dict, batch_size=32, num_workers=0, pin_memory=False)
        assert {"train", "val", "test"} <= set(loaders.keys())

        batch = next(iter(loaders["train"]))
        assert len(batch) == 3
        s1, s2, labels = batch
        assert s1.shape[1] == 20
        assert s2.shape[1] == 15
        assert labels.ndim == 1

    def test_create_loaders_with_extra_sources(self):
        np.random.seed(42)
        n_train, n_val, n_test = 120, 30, 30

        data_dict = {
            "X1_train": np.random.randn(n_train, 10).astype(np.float32),
            "X1_val": np.random.randn(n_val, 10).astype(np.float32),
            "X1_test": np.random.randn(n_test, 10).astype(np.float32),
            "X2_train": np.random.randn(n_train, 8).astype(np.float32),
            "X2_val": np.random.randn(n_val, 8).astype(np.float32),
            "X2_test": np.random.randn(n_test, 8).astype(np.float32),
            "X3_train": np.random.randn(n_train, 6).astype(np.float32),
            "X3_val": np.random.randn(n_val, 6).astype(np.float32),
            "X3_test": np.random.randn(n_test, 6).astype(np.float32),
            "y_train": np.random.randint(0, 4, n_train),
            "y_val": np.random.randint(0, 4, n_val),
            "y_test": np.random.randint(0, 4, n_test),
        }

        loaders = create_multi_source_loaders(data_dict, batch_size=16, num_workers=0, pin_memory=False)
        s1, s2, s3, labels = next(iter(loaders["train"]))

        assert s1.shape[1] == 10
        assert s2.shape[1] == 8
        assert s3.shape[1] == 6
        assert labels.ndim == 1
