#!/usr/bin/env python3
"""
One-click runner for UM-NIDS processed files.

It auto-generates a config for:
1. flow + log feature fusion
2. optional threat-intel decision fusion
3. optional agentic inference mode
"""
import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

import main as project_main
from src.utils.helpers import set_seed


FLOW_KEYWORDS = (
    "flow", "traffic", "net", "pkt", "packet", "byte", "bytes", "octet",
    "src", "dst", "sport", "dport", "port", "proto", "protocol", "tcp",
    "udp", "icmp", "syn", "ack", "fin", "rst", "psh", "urg", "ttl",
    "len", "length", "iat", "duration", "rate", "header", "bulk", "fwd", "bwd",
)

LOG_KEYWORDS = (
    "log", "event", "host", "payload", "context", "message", "alert",
    "status", "error", "warn", "history", "session", "request", "response",
    "http", "dns", "url", "uri", "domain", "query", "user_agent", "agent",
    "command", "process", "service", "registry", "file", "path",
)

LABEL_CANDIDATES = (
    "label", "labels", "class", "attack", "attack_label", "attack_type",
    "target", "y", "category",
)

ID_CANDIDATES = (
    "sample_id", "sampleid", "id", "record_id", "flow_id", "uid", "uuid",
)


def _read_preview(path: str, nrows: int = 2000) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(nrows)
    raise ValueError(f"Unsupported input format for UM-NIDS runner: {suffix}")


def _match_column(columns: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lower_map = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _numeric_feature_columns(frame: pd.DataFrame, excluded: List[str]) -> List[str]:
    excluded_set = {item for item in excluded if item}
    numeric_cols = []
    for column in frame.columns:
        if column in excluded_set:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().any() and series.notna().any():
            numeric_cols.append(column)
    return numeric_cols


def _score_column(column: str, keywords: Tuple[str, ...]) -> int:
    name = str(column).strip().lower()
    return sum(1 for keyword in keywords if keyword in name)


def infer_um_nids_modalities(
    frame: pd.DataFrame,
    label_column: str,
    id_column: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    excluded = [label_column]
    if id_column:
        excluded.append(id_column)
    numeric_cols = _numeric_feature_columns(frame, excluded)
    if len(numeric_cols) < 2:
        raise ValueError("UM-NIDS input must contain at least two numeric feature columns")

    flow_cols = []
    log_cols = []
    for column in numeric_cols:
        flow_score = _score_column(column, FLOW_KEYWORDS)
        log_score = _score_column(column, LOG_KEYWORDS)
        if flow_score > log_score and flow_score > 0:
            flow_cols.append(column)
        elif log_score > flow_score and log_score > 0:
            log_cols.append(column)

    remaining = [column for column in numeric_cols if column not in flow_cols and column not in log_cols]
    if not flow_cols and not log_cols:
        midpoint = max(1, len(numeric_cols) // 2)
        flow_cols = numeric_cols[:midpoint]
        log_cols = numeric_cols[midpoint:]
    elif not flow_cols:
        flow_cols = remaining
        remaining = []
    elif not log_cols:
        log_cols = remaining
        remaining = []
    else:
        for column in remaining:
            if len(flow_cols) <= len(log_cols):
                flow_cols.append(column)
            else:
                log_cols.append(column)

    if not flow_cols or not log_cols:
        raise ValueError(
            "Failed to split UM-NIDS features into flow/log branches. "
            "Pass explicit columns with --flow-columns and --log-columns."
        )
    return flow_cols, log_cols


def _parse_columns(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_runtime_config(
    input_path: str,
    threat_intel_path: Optional[str],
    template_path: str,
    output_config_path: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    label_column: Optional[str],
    id_column: Optional[str],
    flow_columns: List[str],
    log_columns: List[str],
    threat_join_strategy: str,
) -> Tuple[Dict, Dict]:
    config = project_main.load_config(template_path)
    preview = _read_preview(input_path)

    detected_label = label_column or _match_column(preview.columns.tolist(), LABEL_CANDIDATES)
    if detected_label is None:
        raise ValueError("Unable to infer label column. Pass --label-column explicitly.")
    detected_id = id_column or _match_column(preview.columns.tolist(), ID_CANDIDATES)

    resolved_flow_columns = flow_columns
    resolved_log_columns = log_columns
    if not resolved_flow_columns or not resolved_log_columns:
        inferred_flow, inferred_log = infer_um_nids_modalities(preview, detected_label, detected_id)
        resolved_flow_columns = resolved_flow_columns or inferred_flow
        resolved_log_columns = resolved_log_columns or inferred_log

    config["data"]["multimodal"]["enabled"] = True
    config["data"]["multimodal"]["input_format"] = "single_table"
    config["data"]["multimodal"]["path"] = input_path
    config["data"]["multimodal"]["label_column"] = detected_label
    config["data"]["multimodal"]["id_column"] = detected_id
    config["data"]["multimodal"]["flow"]["columns"] = resolved_flow_columns
    config["data"]["multimodal"]["flow"]["prefixes"] = []
    config["data"]["multimodal"]["log"]["columns"] = resolved_log_columns
    config["data"]["multimodal"]["log"]["prefixes"] = []

    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
        config["data"]["loader"]["batch_size"] = batch_size
    if epochs is not None:
        config["training"]["epochs"] = epochs

    threat_summary = {
        "enabled": False,
        "join_strategy": None,
        "feature_columns": [],
    }
    if threat_intel_path:
        intel_preview = _read_preview(threat_intel_path)
        intel_id_column = detected_id if detected_id in intel_preview.columns else None
        if threat_join_strategy == "auto":
            join_strategy = "key" if detected_id and intel_id_column else "row_order"
        else:
            join_strategy = threat_join_strategy
        if join_strategy == "key" and not (detected_id and intel_id_column):
            raise ValueError("Threat-intel key join requires the same id column in both input files")

        excluded = [intel_id_column] if join_strategy == "key" and intel_id_column else []
        intel_feature_columns = _numeric_feature_columns(intel_preview, excluded)
        if not intel_feature_columns:
            raise ValueError("Threat-intel file must contain at least one numeric feature column")

        config["data"]["threat_intel"]["enabled"] = True
        config["data"]["threat_intel"]["source_path"] = threat_intel_path
        config["data"]["threat_intel"]["join_strategy"] = join_strategy
        config["data"]["threat_intel"]["feature_columns"] = intel_feature_columns
        config["model"]["type"] = "decision_fusion_net"
        config["model"]["agentic_mode"]["enabled"] = True

        if join_strategy == "key":
            config["data"]["threat_intel"]["join_key"] = detected_id
            config["data"]["threat_intel"]["intel_key"] = intel_id_column

        threat_summary = {
            "enabled": True,
            "join_strategy": join_strategy,
            "feature_columns": intel_feature_columns,
        }
    else:
        config["data"]["threat_intel"]["enabled"] = False
        config["data"]["threat_intel"]["source_path"] = None
        config["model"]["type"] = "fusion_net"
        config["model"]["agentic_mode"]["enabled"] = False

    output_dir = os.path.dirname(output_config_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    summary = {
        "label_column": detected_label,
        "id_column": detected_id,
        "flow_columns": resolved_flow_columns,
        "log_columns": resolved_log_columns,
        "threat_intel": threat_summary,
        "config_path": output_config_path,
    }
    return config, summary


def main():
    parser = argparse.ArgumentParser(description="One-click runner for UM-NIDS processed files")
    parser.add_argument("--input", required=True, help="Path to processed UM-NIDS CSV/Parquet file")
    parser.add_argument("--threat-intel", default=None, help="Optional threat-intel CSV/Parquet file")
    parser.add_argument("--template-config", default="src/config/config_cicids2018_agentic.yaml", help="Base config template")
    parser.add_argument("--output-config", default=None, help="Generated runtime config path")
    parser.add_argument("--label-column", default=None, help="Override label column")
    parser.add_argument("--id-column", default=None, help="Override id column")
    parser.add_argument("--flow-columns", default=None, help="Comma-separated flow columns override")
    parser.add_argument("--log-columns", default=None, help="Comma-separated log columns override")
    parser.add_argument("--threat-join-strategy", choices=["auto", "key", "row_order"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    threat_intel_path = None
    if args.threat_intel:
        threat_intel_path = os.path.abspath(args.threat_intel)
        if not os.path.exists(threat_intel_path):
            raise FileNotFoundError(f"Threat-intel file not found: {threat_intel_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_config_path = args.output_config or os.path.join(
        "outputs",
        "generated_configs",
        f"um_nids_agentic_{timestamp}.yaml",
    )

    logger = project_main.setup_logging(os.path.join(project_main.project_root, "outputs", "logs"))
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    config, summary = build_runtime_config(
        input_path=input_path,
        threat_intel_path=threat_intel_path,
        template_path=args.template_config,
        output_config_path=output_config_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_column=args.label_column,
        id_column=args.id_column,
        flow_columns=_parse_columns(args.flow_columns),
        log_columns=_parse_columns(args.log_columns),
        threat_join_strategy=args.threat_join_strategy,
    )

    logger.info("UM-NIDS runtime config generated")
    logger.info(f"Config path: {summary['config_path']}")
    logger.info(f"Label column: {summary['label_column']}")
    logger.info(f"ID column: {summary['id_column']}")
    logger.info(f"Flow columns: {len(summary['flow_columns'])}")
    logger.info(f"Log columns: {len(summary['log_columns'])}")
    logger.info(f"Threat-intel enabled: {summary['threat_intel']['enabled']}")
    if summary["threat_intel"]["enabled"]:
        logger.info(f"Threat-intel join: {summary['threat_intel']['join_strategy']}")
        logger.info(f"Threat-intel feature columns: {len(summary['threat_intel']['feature_columns'])}")

    project_main.preprocess_data(None, deepcopy(config), logger)
    _, experiment_name = project_main.train_model(deepcopy(config), logger)
    if experiment_name is None:
        raise RuntimeError("Training did not produce an experiment directory")

    project_main.evaluate_model(deepcopy(config), experiment_name, logger)
    if not args.skip_report:
        project_main.generate_report(deepcopy(config), experiment_name, logger)

    print(f"\nRuntime config: {summary['config_path']}")
    print(f"Experiment: outputs/{experiment_name}")
    print(f"Checkpoint: outputs/{experiment_name}/checkpoints/best_model.pth")
    print(f"Results: outputs/{experiment_name}/results/test_results.pkl")


if __name__ == "__main__":
    main()
