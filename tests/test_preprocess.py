"""
数据预处理相关单元测试
"""
import logging
import pickle

import numpy as np
import pandas as pd

import main as project_main
from src.data.dataloader import CICIDS2017Preprocessor, DataSplitter, ThreatIntelFeatureBuilder


class TestCICPreprocessor:
    """测试 CIC 预处理器"""

    def test_preprocess_dataframe(self):
        """测试直接从 DataFrame 预处理"""
        df = pd.DataFrame({
            'Flow Duration': [10, 20, 30, 40],
            'Total Fwd Packets': [2, 3, 4, 5],
            'SYN Flag Count': [0, 1, 0, 1],
            'Label': ['BENIGN', 'DoS Hulk', 'PortScan', 'BENIGN']
        })

        preprocessor = CICIDS2017Preprocessor()
        result = preprocessor.preprocess_dataframe(
            df=df,
            binary_classification=False,
            feature_selection='all',
            normalize=False
        )

        assert result['X'].shape == (4, 3)
        assert result['y'].shape == (4,)
        assert result['num_features'] == 3
        assert result['num_classes'] == 3

    def test_threat_intel_builder_loads_csv_features(self, tmp_path):
        threat_path = tmp_path / "threat_intel.csv"
        pd.DataFrame({
            "ti_score": [0.1, 0.2, 0.3],
            "ti_match": [1, 0, 1],
        }).to_csv(threat_path, index=False)

        builder = ThreatIntelFeatureBuilder({
            "enabled": True,
            "source_name": "threat_intel",
            "source_path": str(threat_path),
            "join_strategy": "row_order",
        })
        features, names = builder.build_features(n_samples=3)

        assert features.shape == (3, 2)
        assert names == ["ti_score", "ti_match"]

    def test_preprocess_data_appends_threat_intel_as_third_source(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "flows.csv"
        threat_path = tmp_path / "threat_intel.csv"

        df = pd.DataFrame({
            "Flow Duration": np.arange(12) + 1,
            "Total Fwd Packets": np.arange(12) + 10,
            "SYN Flag Count": [0, 1] * 6,
            "Destination Port": [80, 443, 53, 22] * 3,
            "Label": ["BENIGN", "DoS Hulk"] * 6,
        })
        df.to_csv(raw_path, index=False)

        pd.DataFrame({
            "ip_threat_score": np.linspace(0.0, 1.0, 12),
            "signature_count": np.arange(12),
        }).to_csv(threat_path, index=False)

        output_dir = tmp_path / "processed"
        config = {
            "data": {
                "processed_dir": str(output_dir),
                "preprocessing": {
                    "binary_classification": False,
                    "feature_selection": "all",
                    "normalize": False,
                },
                "split": {
                    "test_size": 0.2,
                    "val_size": 0.1,
                    "stratify": True,
                    "random_state": 42,
                },
                "multi_source": {
                    "source1_groups": ["traffic", "temporal"],
                    "source2_groups": ["flags", "header"],
                    "extra_source_groups": [],
                },
                "threat_intel": {
                    "enabled": True,
                    "source_name": "threat_intel",
                    "source_path": str(threat_path),
                    "join_strategy": "row_order",
                },
            }
        }
        logger = logging.getLogger("test-preprocess")
        logger.handlers = []
        logger.addHandler(logging.NullHandler())

        multi_source_data, _ = project_main.preprocess_data(str(raw_dir), config, logger)

        assert multi_source_data["num_sources"] == 3
        assert multi_source_data["source_aliases"][-1] == "threat_intel"
        assert "s3_train" in multi_source_data
        assert multi_source_data["source3_dim"] == 2
        assert multi_source_data["threat_intel"]["enabled"] is True

        with open(output_dir / "multi_source_data.pkl", "rb") as f:
            saved = pickle.load(f)
        assert saved["num_sources"] == 3
        assert saved["source_aliases"][-1] == "threat_intel"


class TestDataSplitter:
    """测试多源划分器"""

    def test_split_multi_source_list(self):
        """测试三个数据源同时划分"""
        np.random.seed(42)
        n = 100
        y = np.random.randint(0, 3, n)
        sources = [
            np.random.randn(n, 5).astype(np.float32),
            np.random.randn(n, 7).astype(np.float32),
            np.random.randn(n, 4).astype(np.float32),
        ]

        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        result = splitter.split_multi_source_list(sources, y, stratify=True)

        for idx in (1, 2, 3):
            assert f'X{idx}_train' in result
            assert f'X{idx}_val' in result
            assert f'X{idx}_test' in result
        assert len(result['y_train']) + len(result['y_val']) + len(result['y_test']) == n
