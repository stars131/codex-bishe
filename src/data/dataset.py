"""
PyTorch dataset utilities for network attack detection.
Supports single-source and multi-source data loading.
"""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class NetworkAttackDataset(Dataset):
    """Single-source tabular dataset."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None,
    ):
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
    """Dataset that returns an arbitrary number of sources plus a label."""

    def __init__(
        self,
        *source_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        source_names: Optional[List[str]] = None,
        transforms: Optional[List[Optional[callable]]] = None,
        source1_name: str = "traffic",
        source2_name: str = "temporal",
        transform1: Optional[callable] = None,
        transform2: Optional[callable] = None,
    ):
        if labels is None:
            if len(source_features) < 3:
                raise ValueError("Expected at least two source arrays and labels")
            *source_arrays, labels = source_features
        else:
            source_arrays = list(source_features)

        if len(source_arrays) < 2:
            raise ValueError("MultiSourceDataset requires at least two source arrays")

        self.sources = [torch.FloatTensor(features) for features in source_arrays]
        self.labels = torch.LongTensor(labels)

        sample_counts = {source.shape[0] for source in self.sources}
        sample_counts.add(self.labels.shape[0])
        if len(sample_counts) != 1:
            raise ValueError("All source arrays and labels must contain the same number of samples")

        if source_names is None:
            source_names = [source1_name, source2_name] + [
                f"source{i}" for i in range(3, len(self.sources) + 1)
            ]
        if len(source_names) < len(self.sources):
            source_names = source_names + [
                f"source{i}" for i in range(len(source_names) + 1, len(self.sources) + 1)
            ]
        self.source_names = source_names[: len(self.sources)]

        if transforms is None:
            transforms = [transform1, transform2] + [None] * max(0, len(self.sources) - 2)
        if len(transforms) < len(self.sources):
            transforms = transforms + [None] * (len(self.sources) - len(transforms))
        self.transforms = transforms[: len(self.sources)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        source_items = [source[idx] for source in self.sources]
        label = self.labels[idx]
        transformed = [
            transform(source_item) if transform else source_item
            for source_item, transform in zip(source_items, self.transforms)
        ]
        return (*transformed, label)

    @property
    def source1_dim(self) -> int:
        return self.sources[0].shape[1]

    @property
    def source2_dim(self) -> int:
        return self.sources[1].shape[1]

    @property
    def source_dims(self) -> List[int]:
        return [source.shape[1] for source in self.sources]

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))


class DataTransforms:
    """Small feature-space augmentations for tabular tensors."""

    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    @staticmethod
    def random_dropout(tensor: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        mask = torch.rand_like(tensor) > p
        return tensor * mask

    @staticmethod
    def feature_scaling(
        tensor: torch.Tensor,
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> torch.Tensor:
        scale = torch.empty(tensor.shape).uniform_(*scale_range)
        return tensor * scale


class TrainingTransform:
    """Composable training-time augmentation."""

    def __init__(
        self,
        noise_std: float = 0.01,
        dropout_p: float = 0.1,
        use_noise: bool = True,
        use_dropout: bool = False,
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
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    augment_train: bool = False,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for single-source data."""
    effective_pin_memory = pin_memory and torch.cuda.is_available()
    train_transform = TrainingTransform(noise_std=0.01) if augment_train else None

    train_dataset = NetworkAttackDataset(
        data_dict["X_train"],
        data_dict["y_train"],
        transform=train_transform,
    )
    val_dataset = NetworkAttackDataset(data_dict["X_val"], data_dict["y_val"])
    test_dataset = NetworkAttackDataset(data_dict["X_test"], data_dict["y_test"])

    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        class_counts = np.bincount(data_dict["y_train"])
        weights = 1.0 / class_counts
        sample_weights = weights[data_dict["y_train"]]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
        ),
    }

    print("\nDataLoader created:")
    print(f"  Train: {len(train_dataset)} samples, {len(loaders['train'])} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(loaders['val'])} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(loaders['test'])} batches")
    return loaders


def create_multi_source_loaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    augment_train: bool = False,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for multi-source data without collapsing sources."""
    effective_pin_memory = pin_memory and torch.cuda.is_available()
    source_indices = sorted({
        int(match.group(1))
        for key in data_dict.keys()
        for match in [re.match(r"^X(\d+)_train$", key)]
        if match is not None
    })

    if not source_indices:
        raise ValueError("No multi-source tensors found in data_dict (expected keys like X1_train, X2_train)")
    if len(source_indices) < 2:
        raise ValueError("Multi-source training requires at least two sources")

    for idx in source_indices:
        for split in ("train", "val", "test"):
            key = f"X{idx}_{split}"
            if key not in data_dict:
                raise KeyError(f"Missing data key: {key}")

    train_transform = TrainingTransform(noise_std=0.01) if augment_train else None
    source_names = [f"source{idx}" for idx in source_indices]

    train_dataset = MultiSourceDataset(
        *[data_dict[f"X{idx}_train"] for idx in source_indices],
        labels=data_dict["y_train"],
        source_names=source_names,
        transforms=[train_transform] * len(source_indices),
    )
    val_dataset = MultiSourceDataset(
        *[data_dict[f"X{idx}_val"] for idx in source_indices],
        labels=data_dict["y_val"],
        source_names=source_names,
    )
    test_dataset = MultiSourceDataset(
        *[data_dict[f"X{idx}_test"] for idx in source_indices],
        labels=data_dict["y_test"],
        source_names=source_names,
    )

    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        class_counts = np.bincount(data_dict["y_train"])
        weights = 1.0 / class_counts
        sample_weights = weights[data_dict["y_train"]]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
        ),
    }

    print("\nMulti-source DataLoader created:")
    print(f"  Train: {len(train_dataset)} samples, {len(loaders['train'])} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(loaders['val'])} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(loaders['test'])} batches")
    for idx, source_dim in enumerate(train_dataset.source_dims, start=1):
        print(f"  Source {idx} dim: {source_dim}")
    return loaders


def get_class_weights(labels: np.ndarray, num_classes: Optional[int] = None) -> torch.Tensor:
    """Compute class weights for loss functions."""
    labels = np.asarray(labels)
    class_counts = np.bincount(labels, minlength=num_classes or 0)
    total = len(labels)
    weights = np.zeros_like(class_counts, dtype=np.float32)
    nonzero_mask = class_counts > 0
    weights[nonzero_mask] = total / (len(class_counts) * class_counts[nonzero_mask])
    return torch.FloatTensor(weights)


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Compute sample weights for weighted sampling."""
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    return weights[labels]
