"""
Tests for one-click UM-NIDS runtime config generation.
"""
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from run_um_nids_agentic import build_runtime_config, infer_um_nids_modalities


def test_infer_um_nids_modalities_splits_flow_and_log_features():
    frame = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(8)],
        "flow_duration": np.arange(8),
        "packet_count": np.arange(8) + 10,
        "payload_entropy": np.arange(8) + 20,
        "event_count": np.arange(8) + 30,
        "label": [0, 1] * 4,
    })

    flow_columns, log_columns = infer_um_nids_modalities(
        frame=frame,
        label_column="label",
        id_column="sample_id",
    )

    assert "flow_duration" in flow_columns
    assert "packet_count" in flow_columns
    assert "payload_entropy" in log_columns
    assert "event_count" in log_columns


def test_build_runtime_config_enables_decision_fusion_when_threat_intel_exists():
    tmp_root = Path("pytest-um-nids-runner")
    tmp_root.mkdir(exist_ok=True)
    tmp_dir = tmp_root / f"case-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = tmp_dir / "um_nids.csv"
        intel_path = tmp_dir / "intel.csv"
        config_path = tmp_dir / "runtime.yaml"

        pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(12)],
            "flow_duration": np.arange(12),
            "packet_count": np.arange(12) + 1,
            "payload_entropy": np.arange(12) + 2,
            "event_count": np.arange(12) + 3,
            "label": [0, 1] * 6,
        }).to_csv(input_path, index=False)

        pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(12)],
            "intel_score": np.linspace(0.1, 0.9, 12),
            "intel_hits": np.arange(12) % 3,
        }).to_csv(intel_path, index=False)

        config, summary = build_runtime_config(
            input_path=str(input_path),
            threat_intel_path=str(intel_path),
            template_path="src/config/config_cicids2018_agentic.yaml",
            output_config_path=str(config_path),
            epochs=5,
            batch_size=16,
            label_column=None,
            id_column=None,
            flow_columns=[],
            log_columns=[],
            threat_join_strategy="auto",
        )

        assert config["model"]["type"] == "decision_fusion_net"
        assert config["model"]["agentic_mode"]["enabled"] is True
        assert config["data"]["threat_intel"]["enabled"] is True
        assert config["data"]["threat_intel"]["join_strategy"] == "key"
        assert config["training"]["epochs"] == 5
        assert config["training"]["batch_size"] == 16
        assert summary["label_column"] == "label"
        assert summary["id_column"] == "sample_id"
        assert "flow_duration" in summary["flow_columns"]
        assert "payload_entropy" in summary["log_columns"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_runtime_config_falls_back_to_two_source_model_without_threat_intel():
    tmp_root = Path("pytest-um-nids-runner")
    tmp_root.mkdir(exist_ok=True)
    tmp_dir = tmp_root / f"case-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = tmp_dir / "um_nids.csv"
        config_path = tmp_dir / "runtime.yaml"

        pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(10)],
            "flow_duration": np.arange(10),
            "packet_count": np.arange(10) + 1,
            "payload_entropy": np.arange(10) + 2,
            "event_count": np.arange(10) + 3,
            "label": [0, 1] * 5,
        }).to_csv(input_path, index=False)

        config, _ = build_runtime_config(
            input_path=str(input_path),
            threat_intel_path=None,
            template_path="src/config/config_cicids2018_agentic.yaml",
            output_config_path=str(config_path),
            epochs=None,
            batch_size=None,
            label_column=None,
            id_column=None,
            flow_columns=[],
            log_columns=[],
            threat_join_strategy="auto",
        )

        assert config["model"]["type"] == "fusion_net"
        assert config["model"]["agentic_mode"]["enabled"] is False
        assert config["data"]["threat_intel"]["enabled"] is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
