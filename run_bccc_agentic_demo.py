#!/usr/bin/env python3
"""
One-click BCCC-CSE-CIC-IDS-2018 sample builder, threat-intel mock API runner, trainer, and evaluator.
"""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

import main as project_main
from run_um_nids_agentic import build_runtime_config
from src.data.bccc_cicids2018 import BCCCCICIDS2018Adapter
from src.threat_intel.mock_api import (
    MockThreatIntelAPIServer,
    ThreatIntelAPIClient,
    ThreatIntelLibraryBuilder,
)
from src.utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run a BCCC-CSE-CIC-IDS-2018 agentic demo")
    parser.add_argument(
        "--dataset-dir",
        default="data/数据集BCCC-CSE-CIC-IDS-2018",
        help="Directory containing the BCCC outer zip archives",
    )
    parser.add_argument(
        "--sample-per-member",
        type=int,
        default=120,
        help="Rows to read from each selected inner archive",
    )
    parser.add_argument(
        "--member-keywords",
        default="benign,bf_ftp,bf_ssh,bot,sql_injection,infiltration",
        help="Comma-separated archive keywords to sample",
    )
    parser.add_argument("--max-members", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    logger = project_main.setup_logging(os.path.join(project_main.project_root, "outputs", "logs"))
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    adapter = BCCCCICIDS2018Adapter(args.dataset_dir)
    member_keywords = [item.strip() for item in args.member_keywords.split(",") if item.strip()]
    raw_df, multimodal_df = adapter.build_multimodal_table(
        sample_per_member=args.sample_per_member,
        keywords=member_keywords,
        max_members=args.max_members,
    )

    processed_dir = Path("data/processed")
    threat_dir = Path("data/threat_intel")
    processed_dir.mkdir(parents=True, exist_ok=True)
    threat_dir.mkdir(parents=True, exist_ok=True)

    raw_path = processed_dir / "bccc_cicids2018_raw_sample.csv"
    multimodal_path = processed_dir / "bccc_cicids2018_agentic_sample.csv"
    adapter.save_outputs(raw_df, multimodal_df, str(raw_path), str(multimodal_path))
    logger.info(f"Saved sampled raw file: {raw_path}")
    logger.info(f"Saved multimodal file: {multimodal_path}")

    library_path = threat_dir / "bccc_cicids2018_mock_library.json"
    intel_path = threat_dir / "bccc_cicids2018_agentic_threat_intel.csv"

    library_builder = ThreatIntelLibraryBuilder()
    library = library_builder.build(raw_df)
    library_builder.save(library, str(library_path))
    logger.info(f"Saved threat-intel library: {library_path}")

    with MockThreatIntelAPIServer(library) as api_server:
        logger.info(f"Mock threat-intel API started at {api_server.base_url}")
        client = ThreatIntelAPIClient(api_server.base_url)
        intel_df = client.enrich_dataframe(raw_df[["sample_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"]])
        intel_df.to_csv(intel_path, index=False)
    logger.info(f"Saved threat-intel sidecar file: {intel_path}")

    config_path = os.path.join("outputs", "generated_configs", "bccc_cicids2018_agentic_runtime.yaml")
    config, summary = build_runtime_config(
        input_path=str(multimodal_path),
        threat_intel_path=str(intel_path),
        template_path="src/config/config_bccc_cicids2018_agentic.yaml",
        output_config_path=config_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_column="label",
        id_column="sample_id",
        flow_columns=[column for column in multimodal_df.columns if column.startswith("flow_")],
        log_columns=[column for column in multimodal_df.columns if column.startswith("log_")],
        threat_join_strategy="key",
    )
    config.setdefault("data", {}).setdefault("loader", {})["num_workers"] = 0
    config["data"]["loader"]["pin_memory"] = False
    config.setdefault("training", {})["mixed_precision"] = False
    logger.info(f"Runtime config saved: {summary['config_path']}")

    project_main.preprocess_data(None, deepcopy(config), logger)
    _, experiment_name = project_main.train_model(deepcopy(config), logger)
    if experiment_name is None:
        raise RuntimeError("Training failed to produce an experiment output directory")
    project_main.evaluate_model(deepcopy(config), experiment_name, logger)
    if not args.skip_report:
        project_main.generate_report(deepcopy(config), experiment_name, logger)

    print(f"\nSample rows: {len(multimodal_df)}")
    print(f"Multimodal data: {multimodal_path}")
    print(f"Threat-intel library: {library_path}")
    print(f"Threat-intel sidecar: {intel_path}")
    print(f"Experiment: outputs/{experiment_name}")
    print(f"Checkpoint: outputs/{experiment_name}/checkpoints/best_model.pth")
    print(f"Results: outputs/{experiment_name}/results/test_results.pkl")


if __name__ == "__main__":
    main()
