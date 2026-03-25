"""
Tests for formal BCCC experiment runner helpers.
"""
import pickle
import shutil
from pathlib import Path

import numpy as np

from src.experiments.formal_bccc import (
    BCCCFormalExperimentRunner,
    default_formal_experiment_specs,
)


def test_default_formal_specs_include_proposed_and_ablation():
    specs = default_formal_experiment_specs()
    names = {spec.name for spec in specs}
    assert "flow_only_single" in names
    assert "flow_log_attention" in names
    assert "flow_log_ti_agentic" in names


def test_formal_runner_builds_source_subsets():
    root = Path("pytest-formal-bccc")
    root.mkdir(exist_ok=True)
    case_dir = root / "case-runner"
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        processed_path = case_dir / "multi_source_data.pkl"
        single_path = case_dir / "single_source_data.pkl"

        data = {
            "s1_train": np.random.randn(12, 4).astype(np.float32),
            "s1_val": np.random.randn(6, 4).astype(np.float32),
            "s1_test": np.random.randn(6, 4).astype(np.float32),
            "s2_train": np.random.randn(12, 3).astype(np.float32),
            "s2_val": np.random.randn(6, 3).astype(np.float32),
            "s2_test": np.random.randn(6, 3).astype(np.float32),
            "s3_train": np.random.randn(12, 2).astype(np.float32),
            "s3_val": np.random.randn(6, 2).astype(np.float32),
            "s3_test": np.random.randn(6, 2).astype(np.float32),
            "y_train": np.array([0, 1] * 6),
            "y_val": np.array([0, 1, 0, 1, 0, 1]),
            "y_test": np.array([0, 1, 0, 1, 0, 1]),
            "class_names": ["benign", "attack"],
            "num_classes": 2,
            "source_aliases": ["flow", "log", "threat_intel"],
        }
        single = {
            "X_train": np.random.randn(12, 7).astype(np.float32),
            "X_val": np.random.randn(6, 7).astype(np.float32),
            "X_test": np.random.randn(6, 7).astype(np.float32),
            "y_train": data["y_train"],
            "y_val": data["y_val"],
            "y_test": data["y_test"],
            "feature_names": [f"f{i}" for i in range(7)],
            "class_names": data["class_names"],
        }
        with open(processed_path, "wb") as f:
            pickle.dump(data, f)
        with open(single_path, "wb") as f:
            pickle.dump(single, f)

        runner = BCCCFormalExperimentRunner(
            config={
                "training": {"batch_size": 4},
                "data": {"loader": {"num_workers": 0, "pin_memory": False, "use_weighted_sampler": False}},
                "model": {"architecture": {"hidden_dim": 16, "dropout": 0.1, "encoder_type": "mlp"}},
            },
            output_dir=str(case_dir / "outputs"),
            processed_data_path=str(processed_path),
            single_source_data_path=str(single_path),
        )

        multi_subset = runner._multi_source_subset([1, 3])
        single_subset = runner._single_source_subset([1, 2])

        assert multi_subset["X1_train"].shape == (12, 4)
        assert multi_subset["X2_train"].shape == (12, 2)
        assert single_subset["X_train"].shape == (12, 7)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
