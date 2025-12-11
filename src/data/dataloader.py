"""
Data loading and preprocessing module for network attack detection.
Supports multi-source data fusion (traffic + logs).
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataLoader:
    """Load and preprocess multi-source network data."""

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_traffic_data(self, path: str) -> pd.DataFrame:
        """Load network traffic data (CSV/PCAP processed)."""
        if path.endswith('.csv'):
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file format: {path}")

    def load_log_data(self, path: str) -> pd.DataFrame:
        """Load system/application log data."""
        if path.endswith('.csv'):
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file format: {path}")

    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """Preprocess features with scaling."""
        if is_training:
            return self.scaler.fit_transform(df)
        return self.scaler.transform(df)

    def encode_labels(self, labels: pd.Series, is_training: bool = True) -> np.ndarray:
        """Encode categorical labels."""
        if is_training:
            return self.label_encoder.fit_transform(labels)
        return self.label_encoder.transform(labels)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, ...]:
        """Split data into train/val/test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


class MultiSourceFusion:
    """Fuse multiple data sources for attack detection."""

    def __init__(self, fusion_method: str = "concat"):
        self.fusion_method = fusion_method

    def fuse(self, traffic_features: np.ndarray, log_features: np.ndarray) -> np.ndarray:
        """Fuse traffic and log features."""
        if self.fusion_method == "concat":
            return np.concatenate([traffic_features, log_features], axis=1)
        raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
