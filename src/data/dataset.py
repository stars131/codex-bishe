"""
PyTorch Dataset classes for network attack detection.
Supports single-source and multi-source data loading.
"""
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class NetworkAttackDataset(Dataset):
    """
    单源网络攻击检测数据集

    用于加载和提供单一数据源的网络流量数据。
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        初始化数据集

        Args:
            features: 特征数组 (N, D)
            labels: 标签数组 (N,)
            transform: 可选的数据转换函数
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))


class MultiSourceDataset(Dataset):
    """
    多源网络攻击检测数据集

    用于加载和提供多个数据源（如流量特征 + 时序特征）的数据。
    支持注意力融合模型的训练。
    """

    def __init__(
        self,
        source1_features: np.ndarray,
        source2_features: np.ndarray,
        labels: np.ndarray,
        source1_name: str = "traffic",
        source2_name: str = "temporal",
        transform1: Optional[callable] = None,
        transform2: Optional[callable] = None
    ):
        """
        初始化多源数据集

        Args:
            source1_features: 第一数据源特征 (N, D1)
            source2_features: 第二数据源特征 (N, D2)
            labels: 标签数组 (N,)
            source1_name: 第一数据源名称
            source2_name: 第二数据源名称
            transform1: 第一数据源的转换函数
            transform2: 第二数据源的转换函数
        """
        self.source1 = torch.FloatTensor(source1_features)
        self.source2 = torch.FloatTensor(source2_features)
        self.labels = torch.LongTensor(labels)

        self.source1_name = source1_name
        self.source2_name = source2_name

        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项

        Returns:
            (source1_features, source2_features, label)
        """
        x1 = self.source1[idx]
        x2 = self.source2[idx]
        y = self.labels[idx]

        if self.transform1:
            x1 = self.transform1(x1)
        if self.transform2:
            x2 = self.transform2(x2)

        return x1, x2, y

    @property
    def source1_dim(self) -> int:
        return self.source1.shape[1]

    @property
    def source2_dim(self) -> int:
        return self.source2.shape[1]

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))


class DataTransforms:
    """
    数据转换工具类

    提供各种数据增强和转换方法。
    """

    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    @staticmethod
    def random_dropout(tensor: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """随机置零（特征dropout）"""
        mask = torch.rand_like(tensor) > p
        return tensor * mask

    @staticmethod
    def feature_scaling(tensor: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """随机特征缩放"""
        scale = torch.empty(tensor.shape).uniform_(*scale_range)
        return tensor * scale


class TrainingTransform:
    """
    训练时的数据增强组合

    可以组合多种增强方法。
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        dropout_p: float = 0.1,
        use_noise: bool = True,
        use_dropout: bool = False
    ):
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.use_noise = use_noise
        self.use_dropout = use_dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_noise and self.noise_std > 0:
            x = DataTransforms.add_gaussian_noise(x, self.noise_std)
        if self.use_dropout and self.dropout_p > 0:
            x = DataTransforms.random_dropout(x, self.dropout_p)
        return x


def create_data_loaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    augment_train: bool = False
) -> Dict[str, DataLoader]:
    """
    创建单源数据的DataLoader

    Args:
        data_dict: 包含训练/验证/测试数据的字典
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_weighted_sampler: 是否使用加权采样（处理类不平衡）
        augment_train: 是否对训练数据进行增强

    Returns:
        包含train/val/test DataLoader的字典
    """
    # 训练数据增强
    train_transform = TrainingTransform(noise_std=0.01) if augment_train else None

    # 创建数据集
    train_dataset = NetworkAttackDataset(
        data_dict['X_train'],
        data_dict['y_train'],
        transform=train_transform
    )
    val_dataset = NetworkAttackDataset(
        data_dict['X_val'],
        data_dict['y_val']
    )
    test_dataset = NetworkAttackDataset(
        data_dict['X_test'],
        data_dict['y_test']
    )

    # 加权采样器（处理类不平衡）
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        class_counts = np.bincount(data_dict['y_train'])
        weights = 1.0 / class_counts
        sample_weights = weights[data_dict['y_train']]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False

    # 创建DataLoader
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    print(f"\nDataLoader 创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(loaders['train'])} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(loaders['val'])} 批次")
    print(f"  测试集: {len(test_dataset)} 样本, {len(loaders['test'])} 批次")

    return loaders


def create_multi_source_loaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    augment_train: bool = False
) -> Dict[str, DataLoader]:
    """
    创建多源数据的DataLoader

    Args:
        data_dict: 包含多源训练/验证/测试数据的字典
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_weighted_sampler: 是否使用加权采样
        augment_train: 是否对训练数据进行增强

    Returns:
        包含train/val/test DataLoader的字典
    """
    # 训练数据增强
    train_transform = TrainingTransform(noise_std=0.01) if augment_train else None

    # 创建数据集
    train_dataset = MultiSourceDataset(
        data_dict['X1_train'],
        data_dict['X2_train'],
        data_dict['y_train'],
        transform1=train_transform,
        transform2=train_transform
    )
    val_dataset = MultiSourceDataset(
        data_dict['X1_val'],
        data_dict['X2_val'],
        data_dict['y_val']
    )
    test_dataset = MultiSourceDataset(
        data_dict['X1_test'],
        data_dict['X2_test'],
        data_dict['y_test']
    )

    # 加权采样器
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        class_counts = np.bincount(data_dict['y_train'])
        weights = 1.0 / class_counts
        sample_weights = weights[data_dict['y_train']]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False

    # 创建DataLoader
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    print(f"\n多源 DataLoader 创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(loaders['train'])} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(loaders['val'])} 批次")
    print(f"  测试集: {len(test_dataset)} 样本, {len(loaders['test'])} 批次")
    print(f"  Source 1 维度: {train_dataset.source1_dim}")
    print(f"  Source 2 维度: {train_dataset.source2_dim}")

    return loaders


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    计算类别权重（用于损失函数）

    Args:
        labels: 标签数组

    Returns:
        类别权重张量
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """
    计算样本权重（用于加权采样）

    Args:
        labels: 标签数组

    Returns:
        样本权重数组
    """
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    return weights[labels]
