#!/usr/bin/env python3
"""
One-click formal experiment runner for BCCC-CSE-CIC-IDS-2018.
"""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import main as project_main
from run_um_nids_agentic import build_runtime_config
from src.data.bccc_cicids2018 import BCCCCICIDS2018Adapter
from src.experiments.formal_bccc import (
    BCCCFormalExperimentRunner,
    default_formal_experiment_specs,
)
from src.threat_intel.mock_api import (
    MockThreatIntelAPIServer,
    ThreatIntelAPIClient,
    ThreatIntelLibraryBuilder,
)
from src.utils.helpers import set_seed


SUITES = {
    "quick": [
        "flow_only_single",
        "log_only_single",
        "flow_log_attention",
        "flow_log_ti_agentic",
    ],
    "full": [spec.name for spec in default_formal_experiment_specs()],
}


def _resolve_suite_specs(suite: str, explicit_names: str):
    specs = default_formal_experiment_specs()
    spec_map = {spec.name: spec for spec in specs}
    if explicit_names:
        names = [item.strip() for item in explicit_names.split(",") if item.strip()]
    else:
        names = SUITES[suite]
    return [spec_map[name] for name in names]


def main():
    parser = argparse.ArgumentParser(description="Run formal BCCC-CSE-CIC-IDS-2018 experiments")
    parser.add_argument("--dataset-dir", default="data/数据集BCCC-CSE-CIC-IDS-2018")
    parser.add_argument("--suite", choices=["quick", "full"], default="full")
    parser.add_argument("--experiments", default="", help="Comma-separated experiment names to override --suite")
    parser.add_argument("--sample-per-member", type=int, default=200)
    parser.add_argument("--max-members", type=int, default=8)
    parser.add_argument(
        "--member-keywords",
        default="benign,bf_ftp,bf_ssh,bot,sql_injection,infiltration",
        help="Comma-separated member keywords. Leave empty to use all discovered members up to --max-members.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path("outputs") / f"formal_bccc_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    logger = project_main.setup_logging(os.path.join(project_main.project_root, "outputs", "logs"))
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Formal experiment output root: {output_root}")

    adapter = BCCCCICIDS2018Adapter(args.dataset_dir)
    member_keywords = [item.strip() for item in args.member_keywords.split(",") if item.strip()]
    if not member_keywords:
        member_keywords = None

    raw_df, multimodal_df = adapter.build_multimodal_table(
        sample_per_member=args.sample_per_member,
        keywords=member_keywords,
        max_members=args.max_members if args.max_members > 0 else None,
    )

    processed_dir = Path("data/processed")
    threat_dir = Path("data/threat_intel")
    processed_dir.mkdir(parents=True, exist_ok=True)
    threat_dir.mkdir(parents=True, exist_ok=True)

    raw_path = processed_dir / "bccc_cicids2018_formal_raw.csv"
    multimodal_path = processed_dir / "bccc_cicids2018_formal_multimodal.csv"
    adapter.save_outputs(raw_df, multimodal_df, str(raw_path), str(multimodal_path))

    library_path = threat_dir / "bccc_cicids2018_formal_mock_library.json"
    intel_path = threat_dir / "bccc_cicids2018_formal_threat_intel.csv"
    library = ThreatIntelLibraryBuilder(strategy="heuristic").build(raw_df)
    ThreatIntelLibraryBuilder.save(library, str(library_path))

    with MockThreatIntelAPIServer(library) as api_server:
        client = ThreatIntelAPIClient(api_server.base_url)
        intel_df = client.enrich_dataframe(raw_df[["sample_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"]])
        intel_df.to_csv(intel_path, index=False)

    runtime_config_path = output_root / "formal_runtime_config.yaml"
    config, _ = build_runtime_config(
        input_path=str(multimodal_path),
        threat_intel_path=str(intel_path),
        template_path="src/config/config_bccc_cicids2018_agentic.yaml",
        output_config_path=str(runtime_config_path),
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
    config["training"]["mixed_precision"] = False

    project_main.preprocess_data(None, deepcopy(config), logger)

    runner = BCCCFormalExperimentRunner(
        config=config,
        output_dir=str(output_root),
        processed_data_path=str(processed_dir / "multi_source_data.pkl"),
        single_source_data_path=str(processed_dir / "single_source_data.pkl"),
    )
    specs = _resolve_suite_specs(args.suite, args.experiments)
    summary_df = runner.run_many(specs)
    summary_df = runner.finalize()

    print(f"\nFormal experiment output: {output_root}")
    print(f"Summary table: {output_root / 'tables' / 'experiment_summary.csv'}")
    print(f"LaTeX table: {output_root / 'tables' / 'experiment_summary.tex'}")
    print(f"Markdown report: {output_root / 'formal_experiment_report.md'}")
    print(f"Dataset report: {output_root / 'dataset_report' / 'bccc_dataset_overview' / 'report.html'}")
    print(f"Experiments run: {len(summary_df)}")


if __name__ == "__main__":
    main()
