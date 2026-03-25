"""
Tests for multimodal flow/log preprocessing and decision-level fusion.
"""
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.multimodal_builder import MultimodalProcessedDataBuilder
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.models.fusion_net import create_model


def test_multimodal_builder_supports_key_joined_threat_intel():
    tmp_root = Path("pytest-multimodal-builder")
    tmp_root.mkdir(exist_ok=True)
    tmp_dir = tmp_root / f"case-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    table_path = tmp_dir / "um_nids.csv"
    intel_path = tmp_dir / "threat_intel.csv"

    try:
        frame = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(12)],
            "flow_bytes": np.arange(12),
            "flow_pkts": np.arange(12) + 100,
            "log_event_count": np.arange(12) + 200,
            "log_error_count": np.arange(12) + 300,
            "label": [0, 1] * 6,
        })
        frame.to_csv(table_path, index=False)

        intel = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(12)],
            "intel_score": np.linspace(0.1, 0.9, 12),
            "intel_hits": np.arange(12) % 3,
        })
        intel.to_csv(intel_path, index=False)

        builder = MultimodalProcessedDataBuilder({
            "data": {
                "multimodal": {
                    "enabled": True,
                    "input_format": "single_table",
                    "path": str(table_path),
                    "label_column": "label",
                    "id_column": "sample_id",
                    "flow": {"name": "flow", "prefixes": ["flow_"]},
                    "log": {"name": "log", "prefixes": ["log_"]},
                },
                "threat_intel": {
                    "enabled": True,
                    "source_name": "threat_intel",
                    "source_path": str(intel_path),
                    "join_strategy": "key",
                    "join_key": "sample_id",
                    "intel_key": "sample_id",
                },
                "split": {
                    "test_size": 0.2,
                    "val_size": 0.1,
                    "stratify": True,
                    "random_state": 42,
                },
            },
        })

        multi_source_data, single_source_data = builder.build()

        assert multi_source_data["num_sources"] == 3
        assert multi_source_data["source_aliases"] == ["flow", "log", "threat_intel"]
        assert multi_source_data["source3_dim"] == 2
        assert "s3_train" in multi_source_data
        assert single_source_data["num_features"] == 4
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_multimodal_builder_presplit_accepts_single_column_label_files():
    tmp_root = Path("pytest-multimodal-builder")
    tmp_root.mkdir(exist_ok=True)
    tmp_dir = tmp_root / f"case-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for split_name, start in [("train", 0), ("val", 20), ("test", 40)]:
            flow = pd.DataFrame({
                "flow_a": np.arange(start, start + 6),
                "flow_b": np.arange(start, start + 6) + 1,
            })
            log = pd.DataFrame({
                "log_a": np.arange(start, start + 6) + 10,
                "log_b": np.arange(start, start + 6) + 11,
            })
            labels = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1]})
            ids = pd.DataFrame({"sample_id": [f"{split_name}_{i}" for i in range(6)]})

            flow.to_csv(tmp_dir / f"{split_name}_flow.csv", index=False)
            log.to_csv(tmp_dir / f"{split_name}_log.csv", index=False)
            labels.to_csv(tmp_dir / f"{split_name}_label.csv", index=False)
            ids.to_csv(tmp_dir / f"{split_name}_ids.csv", index=False)

        builder = MultimodalProcessedDataBuilder({
            "data": {
                "multimodal": {
                    "enabled": True,
                    "input_format": "pre_split",
                    "splits": {
                        split_name: {
                            "flow_path": str(tmp_dir / f"{split_name}_flow.csv"),
                            "log_path": str(tmp_dir / f"{split_name}_log.csv"),
                            "label_path": str(tmp_dir / f"{split_name}_label.csv"),
                            "id_path": str(tmp_dir / f"{split_name}_ids.csv"),
                        }
                        for split_name in ["train", "val", "test"]
                    },
                },
                "split": {
                    "test_size": 0.2,
                    "val_size": 0.1,
                    "stratify": True,
                    "random_state": 42,
                },
            },
        })

        multi_source_data, _ = builder.build()

        assert multi_source_data["s1_train"].shape == (6, 2)
        assert multi_source_data["s2_test"].shape == (6, 2)
        assert multi_source_data["y_val"].shape == (6,)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_multimodal_builder_rejects_non_numeric_threat_intel_features():
    tmp_root = Path("pytest-multimodal-builder")
    tmp_root.mkdir(exist_ok=True)
    tmp_dir = tmp_root / f"case-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    table_path = tmp_dir / "um_nids.csv"
    intel_path = tmp_dir / "threat_intel.csv"

    try:
        pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(6)],
            "flow_bytes": np.arange(6),
            "flow_pkts": np.arange(6) + 100,
            "log_event_count": np.arange(6) + 200,
            "log_error_count": np.arange(6) + 300,
            "label": [0, 1, 0, 1, 0, 1],
        }).to_csv(table_path, index=False)

        pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(6)],
            "intel_text": ["ioc"] * 6,
        }).to_csv(intel_path, index=False)

        builder = MultimodalProcessedDataBuilder({
            "data": {
                "multimodal": {
                    "enabled": True,
                    "input_format": "single_table",
                    "path": str(table_path),
                    "label_column": "label",
                    "id_column": "sample_id",
                    "flow": {"prefixes": ["flow_"]},
                    "log": {"prefixes": ["log_"]},
                },
                "threat_intel": {
                    "enabled": True,
                    "source_path": str(intel_path),
                    "join_strategy": "key",
                    "join_key": "sample_id",
                    "intel_key": "sample_id",
                },
            },
        })

        try:
            builder.build()
            assert False, "Expected non-numeric threat-intel columns to raise"
        except ValueError as exc:
            assert "non-numeric columns" in str(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_decision_fusion_net_forward_supports_agentic_mode(sample_dims):
    batch_size = sample_dims["batch_size"]
    source_dims = [sample_dims["source1_dim"], sample_dims["source2_dim"], 6]

    model = create_model(
        model_type="decision_fusion_net",
        traffic_dim=source_dims[0],
        log_dim=source_dims[1],
        num_classes=sample_dims["num_classes"],
        config={
            "hidden_dim": sample_dims["hidden_dim"],
            "dropout": 0.1,
            "encoder_type": "mlp",
            "fusion_type": "attention",
            "num_layers": 2,
            "num_heads": 4,
            "source_dims": source_dims,
            "agentic_mode": {
                "enabled": True,
                "uncertainty_threshold": 0.99,
                "intel_confidence_threshold": 0.0,
            },
        },
    )
    model.eval()

    with torch.no_grad():
        logits, attention = model(
            torch.randn(batch_size, source_dims[0]),
            torch.randn(batch_size, source_dims[1]),
            torch.randn(batch_size, source_dims[2]),
        )

    assert logits.shape == (batch_size, sample_dims["num_classes"])
    assert attention.shape == (batch_size, 3)
    assert len(model.last_agentic_actions) == batch_size


def test_evaluator_collects_agentic_actions(sample_dims):
    source_dims = [sample_dims["source1_dim"], sample_dims["source2_dim"], 6]
    model = create_model(
        model_type="decision_fusion_net",
        traffic_dim=source_dims[0],
        log_dim=source_dims[1],
        num_classes=sample_dims["num_classes"],
        config={
            "hidden_dim": sample_dims["hidden_dim"],
            "encoder_type": "mlp",
            "fusion_type": "attention",
            "source_dims": source_dims,
            "agentic_mode": {
                "enabled": True,
                "uncertainty_threshold": 0.99,
                "intel_confidence_threshold": 0.0,
            },
        },
    )
    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.randn(10, source_dims[0]),
            torch.randn(10, source_dims[1]),
            torch.randn(10, source_dims[2]),
            torch.randint(0, sample_dims["num_classes"], (10,)),
        ),
        batch_size=5,
        shuffle=False,
    )

    evaluator = ComprehensiveEvaluator(
        model=model,
        device=torch.device("cpu"),
        class_names=[str(i) for i in range(sample_dims["num_classes"])],
    )
    predictions = evaluator.predict(loader)

    assert predictions["attention_weights"].shape == (10, 3)
    assert predictions["agentic_actions"].shape == (10,)
