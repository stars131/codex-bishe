"""
Utilities for building flow/log/threat-intel datasets from processed files.
"""
import os
import pickle
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.dataloader import DataSplitter


def _load_table(path: str):
    if path is None:
        raise ValueError("Input path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    raise ValueError(f"Unsupported input file format: {suffix}")


def _to_dataframe(payload, name: str) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, dict):
        if "X" in payload:
            feature_names = payload.get("feature_names")
            return pd.DataFrame(payload["X"], columns=feature_names)
        return pd.DataFrame(payload)
    if isinstance(payload, np.ndarray):
        if payload.ndim != 2:
            raise ValueError(f"{name} array must be 2D, got shape {payload.shape}")
        return pd.DataFrame(payload)
    raise TypeError(f"Unsupported payload type for {name}: {type(payload).__name__}")


def _to_array(payload, name: str) -> np.ndarray:
    if isinstance(payload, pd.DataFrame):
        if payload.shape[1] == 1:
            return payload.iloc[:, 0].to_numpy()
        return payload.to_numpy(dtype=np.float32)
    if isinstance(payload, pd.Series):
        return payload.to_numpy()
    if isinstance(payload, dict):
        if "X" in payload:
            return np.asarray(payload["X"], dtype=np.float32)
        if "y" in payload:
            return np.asarray(payload["y"])
        raise ValueError(f"Unsupported dict payload for {name}; expected X or y key")
    if isinstance(payload, np.ndarray):
        return np.asarray(payload)
    if isinstance(payload, list):
        return np.asarray(payload)
    raise TypeError(f"Unsupported payload type for {name}: {type(payload).__name__}")


def _frame_to_float_array(frame: pd.DataFrame, name: str) -> np.ndarray:
    converted = frame.apply(pd.to_numeric, errors="coerce")
    invalid_columns = [
        column
        for column in frame.columns
        if frame[column].notna().any() and converted[column].isna().all()
    ]
    if invalid_columns:
        preview = invalid_columns[:10]
        raise ValueError(
            f"{name} contains non-numeric columns that cannot be used for model input: {preview}"
        )
    values = converted.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return values


def _resolve_columns(frame: pd.DataFrame, spec: Dict, default_exclude: Optional[List[str]] = None) -> List[str]:
    default_exclude = default_exclude or []
    columns = spec.get("columns")
    if columns:
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Columns missing from input table: {missing}")
        return list(columns)

    prefixes = spec.get("prefixes") or []
    regexes = spec.get("regex") or []
    selected = []
    for col in frame.columns:
        if col in default_exclude:
            continue
        if prefixes and any(str(col).startswith(prefix) for prefix in prefixes):
            selected.append(col)
            continue
        if regexes and any(pd.Series([str(col)]).str.contains(pattern, regex=True, na=False).iloc[0] for pattern in regexes):
            selected.append(col)
    return selected


class MultimodalProcessedDataBuilder:
    """Build standard project artifacts from processed flow/log tables."""

    def __init__(self, config: Dict):
        self.config = deepcopy(config)
        self.data_config = self.config.get("data", {})
        self.mm_config = self.data_config.get("multimodal", {})
        self.ti_config = self.data_config.get("threat_intel", {})
        split_config = self.data_config.get("split", {})
        self.splitter = DataSplitter(
            test_size=split_config.get("test_size", 0.2),
            val_size=split_config.get("val_size", 0.1),
            random_state=split_config.get("random_state", 42),
        )

    def build(self) -> Tuple[Dict, Dict]:
        if not self.mm_config.get("enabled", False):
            raise ValueError("data.multimodal.enabled must be true to use MultimodalProcessedDataBuilder")

        input_format = self.mm_config.get("input_format", "single_table")
        if input_format == "single_table":
            return self._build_from_single_table()
        if input_format == "pre_split":
            return self._build_from_pre_split()
        raise ValueError(f"Unsupported multimodal input_format: {input_format}")

    @staticmethod
    def _sanitize_array(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def _maybe_normalize_split_data(self, split_data: Dict[str, np.ndarray], num_sources: int) -> Dict[str, np.ndarray]:
        if not self.data_config.get("preprocessing", {}).get("normalize", False):
            for idx in range(1, num_sources + 1):
                for split_name in ("train", "val", "test"):
                    key = f"X{idx}_{split_name}"
                    split_data[key] = self._sanitize_array(split_data[key])
            return split_data

        for idx in range(1, num_sources + 1):
            scaler = StandardScaler()
            train_key = f"X{idx}_train"
            val_key = f"X{idx}_val"
            test_key = f"X{idx}_test"

            train_values = self._sanitize_array(split_data[train_key])
            val_values = self._sanitize_array(split_data[val_key])
            test_values = self._sanitize_array(split_data[test_key])

            split_data[train_key] = scaler.fit_transform(train_values).astype(np.float32)
            split_data[val_key] = scaler.transform(val_values).astype(np.float32)
            split_data[test_key] = scaler.transform(test_values).astype(np.float32)

        return split_data

    def _extract_modalities_from_table(
        self,
        frame: pd.DataFrame,
        split_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        label_column = self.mm_config.get("label_column", "label")
        if label_column not in frame.columns:
            raise ValueError(f"Label column '{label_column}' not found in multimodal input")

        id_column = self.mm_config.get("id_column")
        reserved = [label_column]
        if id_column and id_column in frame.columns:
            reserved.append(id_column)

        flow_spec = self.mm_config.get("flow", {})
        log_spec = self.mm_config.get("log", {})
        flow_columns = _resolve_columns(frame, flow_spec, default_exclude=reserved)
        log_columns = _resolve_columns(frame, log_spec, default_exclude=reserved)
        if not flow_columns:
            raise ValueError("No flow columns resolved from multimodal input")
        if not log_columns:
            raise ValueError("No log columns resolved from multimodal input")

        result = {
            "flow": _frame_to_float_array(frame[flow_columns], "flow features"),
            "log": _frame_to_float_array(frame[log_columns], "log features"),
            "labels": frame[label_column].to_numpy(),
            "flow_columns": flow_columns,
            "log_columns": log_columns,
        }
        if id_column and id_column in frame.columns:
            result["ids"] = frame[id_column].to_numpy()

        if self.ti_config.get("enabled", False):
            ti_features, ti_names = self._build_threat_intel_features(
                split_name=split_name,
                n_samples=len(frame),
                ids=result.get("ids"),
            )
            result["threat_intel"] = ti_features
            result["threat_intel_columns"] = ti_names

        return result

    def _build_threat_intel_features(
        self,
        split_name: Optional[str],
        n_samples: int,
        ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        split_paths = self.ti_config.get("split_paths") or {}
        source_path = split_paths.get(split_name) if split_name else None
        source_path = source_path or self.ti_config.get("source_path")
        if not source_path:
            raise ValueError("Threat-intel source_path or split_paths must be configured when enabled")

        payload = _load_table(source_path)
        frame = _to_dataframe(payload, "threat_intel")
        join_strategy = self.ti_config.get("join_strategy", "row_order")
        feature_columns = self.ti_config.get("feature_columns")
        if feature_columns:
            missing = [col for col in feature_columns if col not in frame.columns]
            if missing:
                raise ValueError(f"Threat-intel feature columns missing: {missing}")
            feature_frame = frame[feature_columns].copy()
        else:
            excluded = set()
            if join_strategy == "key":
                excluded.add(self.ti_config.get("intel_key", self.ti_config.get("join_key", "sample_id")))
            feature_frame = frame[[col for col in frame.columns if col not in excluded]].copy()
        if feature_frame.empty:
            raise ValueError("Threat-intel feature set is empty; configure feature_columns or provide numeric feature fields")

        if join_strategy == "row_order":
            if len(feature_frame) != n_samples:
                raise ValueError(
                    f"Threat-intel row count ({len(feature_frame)}) does not match multimodal row count ({n_samples})"
                )
            return _frame_to_float_array(feature_frame, "threat-intel features"), list(feature_frame.columns)

        if join_strategy == "key":
            join_key = self.ti_config.get("join_key") or self.mm_config.get("id_column")
            intel_key = self.ti_config.get("intel_key", join_key)
            if ids is None:
                raise ValueError("Threat-intel key-based join requires an id_column in multimodal input")
            if intel_key not in frame.columns:
                raise ValueError(f"Threat-intel key column '{intel_key}' not found")
            intel_frame = frame[[intel_key, *feature_frame.columns]].copy()
            intel_frame = intel_frame.drop_duplicates(subset=[intel_key])
            merged = pd.DataFrame({join_key: ids}).merge(
                intel_frame,
                left_on=join_key,
                right_on=intel_key,
                how="left",
            )
            merged = merged.drop(columns=[join_key, intel_key], errors="ignore").fillna(0.0)
            return _frame_to_float_array(merged, "threat-intel features"), list(feature_frame.columns)

        raise ValueError(f"Unsupported threat-intel join_strategy: {join_strategy}")

    def _build_from_single_table(self) -> Tuple[Dict, Dict]:
        input_path = self.mm_config.get("path")
        if not input_path:
            raise ValueError("data.multimodal.path is required for single_table input")

        table = _to_dataframe(_load_table(input_path), "multimodal")
        extracted = self._extract_modalities_from_table(table)
        encoded_labels, class_names = self._encode_labels(extracted["labels"])

        source_arrays = [extracted["flow"], extracted["log"]]
        source_names = [
            self.mm_config.get("flow", {}).get("name", "flow"),
            self.mm_config.get("log", {}).get("name", "log"),
        ]
        source_feature_names = [extracted["flow_columns"], extracted["log_columns"]]
        if "threat_intel" in extracted:
            source_arrays.append(extracted["threat_intel"])
            source_names.append(self.ti_config.get("source_name", "threat_intel"))
            source_feature_names.append(extracted["threat_intel_columns"])

        split_data = self.splitter.split_multi_source_list(
            source_arrays,
            encoded_labels,
            stratify=self.data_config.get("split", {}).get("stratify", True),
        )
        split_data = self._maybe_normalize_split_data(split_data, len(source_arrays))
        return self._package_outputs(
            split_data=split_data,
            source_arrays=source_arrays,
            source_feature_names=source_feature_names,
            source_aliases=source_names,
            class_names=class_names,
            source_ids=None,
        )

    def _extract_split_arrays(self, split_spec: Dict, split_name: str) -> Dict[str, np.ndarray]:
        if "path" in split_spec:
            frame = _to_dataframe(_load_table(split_spec["path"]), f"{split_name}_table")
            return self._extract_modalities_from_table(frame, split_name=split_name)

        flow = _to_array(_load_table(split_spec["flow_path"]), f"{split_name}_flow").astype(np.float32)
        log = _to_array(_load_table(split_spec["log_path"]), f"{split_name}_log").astype(np.float32)
        labels = _to_array(_load_table(split_spec["label_path"]), f"{split_name}_labels")

        if len(flow) != len(log) or len(flow) != len(labels):
            raise ValueError(f"Split '{split_name}' has inconsistent sample counts across flow/log/labels")

        result = {
            "flow": flow,
            "log": log,
            "labels": labels,
            "flow_columns": split_spec.get("flow_columns") or [f"flow_{i + 1}" for i in range(flow.shape[1])],
            "log_columns": split_spec.get("log_columns") or [f"log_{i + 1}" for i in range(log.shape[1])],
        }
        id_path = split_spec.get("id_path")
        if id_path:
            result["ids"] = _to_array(_load_table(id_path), f"{split_name}_ids")

        if self.ti_config.get("enabled", False):
            ti_features, ti_names = self._build_threat_intel_features(
                split_name=split_name,
                n_samples=len(labels),
                ids=result.get("ids"),
            )
            result["threat_intel"] = ti_features
            result["threat_intel_columns"] = ti_names
        return result

    def _build_from_pre_split(self) -> Tuple[Dict, Dict]:
        split_specs = self.mm_config.get("splits", {})
        required = ["train", "val", "test"]
        missing = [name for name in required if name not in split_specs]
        if missing:
            raise ValueError(f"Missing multimodal split definitions: {missing}")

        split_payloads = {name: self._extract_split_arrays(split_specs[name], name) for name in required}
        train_payload = split_payloads["train"]
        all_labels = np.concatenate([split_payloads[name]["labels"] for name in required])
        _, class_names = self._encode_labels(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        for name in required:
            split_payloads[name]["labels"] = self._transform_labels(split_payloads[name]["labels"], label_to_idx)

        source_aliases = [
            self.mm_config.get("flow", {}).get("name", "flow"),
            self.mm_config.get("log", {}).get("name", "log"),
        ]
        source_feature_names = [train_payload["flow_columns"], train_payload["log_columns"]]
        source_count = 2
        if "threat_intel" in train_payload:
            source_aliases.append(self.ti_config.get("source_name", "threat_intel"))
            source_feature_names.append(train_payload["threat_intel_columns"])
            source_count = 3

        split_data = {
            "y_train": split_payloads["train"]["labels"],
            "y_val": split_payloads["val"]["labels"],
            "y_test": split_payloads["test"]["labels"],
        }
        for idx, key in enumerate(["flow", "log"], start=1):
            split_data[f"X{idx}_train"] = split_payloads["train"][key]
            split_data[f"X{idx}_val"] = split_payloads["val"][key]
            split_data[f"X{idx}_test"] = split_payloads["test"][key]
        if source_count == 3:
            split_data["X3_train"] = split_payloads["train"]["threat_intel"]
            split_data["X3_val"] = split_payloads["val"]["threat_intel"]
            split_data["X3_test"] = split_payloads["test"]["threat_intel"]
        split_data = self._maybe_normalize_split_data(split_data, source_count)

        source_arrays = [split_payloads["train"]["flow"], split_payloads["train"]["log"]]
        if source_count == 3:
            source_arrays.append(split_payloads["train"]["threat_intel"])
        return self._package_outputs(
            split_data=split_data,
            source_arrays=source_arrays,
            source_feature_names=source_feature_names,
            source_aliases=source_aliases,
            class_names=class_names,
            source_ids=None,
        )

    def _class_names(self, labels: np.ndarray) -> List[str]:
        labels = np.asarray(labels)
        if labels.dtype.kind in {"U", "S", "O"}:
            unique = list(dict.fromkeys(labels.tolist()))
            return [str(item) for item in unique]
        unique = sorted(np.unique(labels).tolist())
        return [str(item) for item in unique]

    def _encode_labels(self, labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        labels = np.asarray(labels)
        if labels.dtype.kind in {"U", "S", "O"}:
            class_names = self._class_names(labels)
            label_to_idx = {label: idx for idx, label in enumerate(class_names)}
            encoded = self._transform_labels(labels, label_to_idx)
            return encoded, class_names
        labels = labels.astype(np.int64)
        return labels, self._class_names(labels)

    @staticmethod
    def _transform_labels(labels: np.ndarray, label_to_idx: Dict) -> np.ndarray:
        return np.asarray([label_to_idx[str(label)] if str(label) in label_to_idx else label_to_idx[label] for label in labels], dtype=np.int64)

    def _package_outputs(
        self,
        split_data: Dict[str, np.ndarray],
        source_arrays: List[np.ndarray],
        source_feature_names: List[List[str]],
        source_aliases: List[str],
        class_names: List[str],
        source_ids: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[Dict, Dict]:
        multi_source_data = {
            "y_train": split_data["y_train"],
            "y_val": split_data["y_val"],
            "y_test": split_data["y_test"],
            "class_names": class_names,
            "num_classes": len(class_names),
            "num_sources": len(source_arrays),
            "source_aliases": source_aliases,
            "source_ids": source_ids,
            "input_format": self.mm_config.get("input_format", "single_table"),
        }
        for idx, feature_names in enumerate(source_feature_names, start=1):
            multi_source_data[f"s{idx}_train"] = split_data[f"X{idx}_train"]
            multi_source_data[f"s{idx}_val"] = split_data[f"X{idx}_val"]
            multi_source_data[f"s{idx}_test"] = split_data[f"X{idx}_test"]
            multi_source_data[f"source{idx}_names"] = feature_names
            multi_source_data[f"source{idx}_dim"] = source_arrays[idx - 1].shape[1]
        multi_source_data["source1_names"] = source_feature_names[0]
        multi_source_data["source2_names"] = source_feature_names[1]
        multi_source_data["source1_dim"] = source_arrays[0].shape[1]
        multi_source_data["source2_dim"] = source_arrays[1].shape[1]

        single_train = np.concatenate([split_data["X1_train"], split_data["X2_train"]], axis=1)
        single_val = np.concatenate([split_data["X1_val"], split_data["X2_val"]], axis=1)
        single_test = np.concatenate([split_data["X1_test"], split_data["X2_test"]], axis=1)
        single_source_data = {
            "X_train": single_train,
            "X_val": single_val,
            "X_test": single_test,
            "y_train": split_data["y_train"],
            "y_val": split_data["y_val"],
            "y_test": split_data["y_test"],
            "feature_names": source_feature_names[0] + source_feature_names[1],
            "class_names": class_names,
            "num_features": single_train.shape[1],
            "num_classes": len(class_names),
        }
        return multi_source_data, single_source_data
