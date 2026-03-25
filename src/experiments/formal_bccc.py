"""
Formal experiment runner for BCCC-CSE-CIC-IDS-2018.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

from src.data.dataset import create_data_loaders, create_multi_source_loaders
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.models.fusion_net import create_model
from src.train import Trainer, setup_logger
from src.visualization.report import ExperimentReport

logging.getLogger("matplotlib").setLevel(logging.ERROR)


@dataclass
class ExperimentSpec:
    name: str
    family: str
    model_type: str
    source_indices: List[int]
    fusion_type: Optional[str] = None
    encoder_type: Optional[str] = None
    use_agentic: bool = False
    notes: str = ""


def default_formal_experiment_specs() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="flow_only_single",
            family="ablation",
            model_type="single_source",
            source_indices=[1],
            notes="Single-source baseline using only flow features.",
        ),
        ExperimentSpec(
            name="log_only_single",
            family="ablation",
            model_type="single_source",
            source_indices=[2],
            notes="Single-source baseline using only derived log/context features.",
        ),
        ExperimentSpec(
            name="flow_log_attention",
            family="fusion",
            model_type="fusion_net",
            source_indices=[1, 2],
            fusion_type="attention",
            notes="Two-source feature fusion baseline.",
        ),
        ExperimentSpec(
            name="flow_log_gated",
            family="fusion",
            model_type="fusion_net",
            source_indices=[1, 2],
            fusion_type="gated",
            notes="Two-source gated fusion variant.",
        ),
        ExperimentSpec(
            name="flow_log_multi_head",
            family="fusion",
            model_type="fusion_net",
            source_indices=[1, 2],
            fusion_type="multi_head",
            notes="Two-source multi-head fusion variant.",
        ),
        ExperimentSpec(
            name="flow_log_ti_decision",
            family="proposed",
            model_type="decision_fusion_net",
            source_indices=[1, 2, 3],
            fusion_type="attention",
            use_agentic=False,
            notes="Decision-level fusion with threat-intel but without agentic controller.",
        ),
        ExperimentSpec(
            name="flow_log_ti_agentic",
            family="proposed",
            model_type="decision_fusion_net",
            source_indices=[1, 2, 3],
            fusion_type="attention",
            use_agentic=True,
            notes="Proposed model with flow/log fusion, threat-intel decision fusion, and agentic inference.",
        ),
    ]


class BCCCFormalExperimentRunner:
    def __init__(
        self,
        config: Dict,
        output_dir: str,
        processed_data_path: str,
        single_source_data_path: Optional[str] = None,
    ):
        self.base_config = copy.deepcopy(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_path = processed_data_path
        self.single_source_data_path = single_source_data_path

        with open(processed_data_path, "rb") as f:
            self.data = pickle.load(f)

        self.class_names = self.data["class_names"]
        self.num_classes = int(self.data["num_classes"])
        self.source_aliases = self.data.get("source_aliases", [])
        self.source_indices = sorted(
            int(key[1:key.index("_")])
            for key in self.data.keys()
            if key.startswith("s") and key.endswith("_train")
        )
        self.results: List[Dict] = []
        self.predictions: Dict[str, np.ndarray] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _multi_source_subset(self, source_indices: Sequence[int]) -> Dict[str, np.ndarray]:
        subset = {
            "y_train": self.data["y_train"],
            "y_val": self.data["y_val"],
            "y_test": self.data["y_test"],
        }
        for new_idx, old_idx in enumerate(source_indices, start=1):
            for split in ("train", "val", "test"):
                subset[f"X{new_idx}_{split}"] = self.data[f"s{old_idx}_{split}"]
        return subset

    def _single_source_subset(self, source_indices: Sequence[int]) -> Dict[str, np.ndarray]:
        subset = {
            "y_train": self.data["y_train"],
            "y_val": self.data["y_val"],
            "y_test": self.data["y_test"],
        }
        for split in ("train", "val", "test"):
            arrays = [self.data[f"s{idx}_{split}"] for idx in source_indices]
            subset[f"X_{split}"] = np.concatenate(arrays, axis=1).astype(np.float32)
        return {
            "X_train": subset["X_train"],
            "X_val": subset["X_val"],
            "X_test": subset["X_test"],
            "y_train": subset["y_train"],
            "y_val": subset["y_val"],
            "y_test": subset["y_test"],
        }

    def _infer_source_dims(self, source_indices: Sequence[int]) -> List[int]:
        return [int(self.data[f"s{idx}_train"].shape[1]) for idx in source_indices]

    def _build_config_for_spec(self, spec: ExperimentSpec, source_dims: List[int]) -> Tuple[Dict, Dict]:
        config = copy.deepcopy(self.base_config)
        config.setdefault("model", {})
        config["model"]["type"] = spec.model_type
        config["model"]["source_dims"] = list(source_dims)
        config["model"]["source1_dim"] = source_dims[0]
        config["model"]["source2_dim"] = source_dims[1] if len(source_dims) > 1 else source_dims[0]
        config["model"]["num_classes"] = self.num_classes

        if spec.model_type == "decision_fusion_net":
            config["model"].setdefault("agentic_mode", {})
            config["model"]["agentic_mode"]["enabled"] = bool(spec.use_agentic)
        elif "agentic_mode" in config.get("model", {}):
            config["model"]["agentic_mode"]["enabled"] = False

        factory_config = dict(config.get("model", {}).get("architecture", {}))
        factory_config["source_dims"] = list(source_dims)
        factory_config["fusion_type"] = spec.fusion_type or config.get("model", {}).get("fusion", {}).get("method", "attention")
        factory_config["encoder_type"] = spec.encoder_type or factory_config.get("encoder_type", "mlp")
        factory_config["decision_fusion"] = copy.deepcopy(config.get("model", {}).get("decision_fusion", {}))
        factory_config["agentic_mode"] = copy.deepcopy(config.get("model", {}).get("agentic_mode", {}))
        return config, factory_config

    def _create_loaders(self, spec: ExperimentSpec):
        loader_config = self.base_config.get("data", {}).get("loader", {})
        if spec.model_type == "single_source":
            data_dict = self._single_source_subset(spec.source_indices)
            return create_data_loaders(
                data_dict,
                batch_size=self.base_config["training"]["batch_size"],
                num_workers=loader_config.get("num_workers", 0),
                pin_memory=loader_config.get("pin_memory", self.device.type == "cuda"),
                use_weighted_sampler=loader_config.get("use_weighted_sampler", False),
                augment_train=loader_config.get("augment_train", False),
            )

        data_dict = self._multi_source_subset(spec.source_indices)
        return create_multi_source_loaders(
            data_dict,
            batch_size=self.base_config["training"]["batch_size"],
            num_workers=loader_config.get("num_workers", 0),
            pin_memory=loader_config.get("pin_memory", self.device.type == "cuda"),
            use_weighted_sampler=loader_config.get("use_weighted_sampler", False),
            augment_train=loader_config.get("augment_train", False),
        )

    def _build_model(self, spec: ExperimentSpec, factory_config: Dict, source_dims: List[int]):
        if spec.model_type == "single_source":
            traffic_dim = int(sum(source_dims))
            log_dim = source_dims[0] if source_dims else traffic_dim
        else:
            traffic_dim = source_dims[0]
            log_dim = source_dims[1] if len(source_dims) > 1 else source_dims[0]

        return create_model(
            model_type=spec.model_type,
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            num_classes=self.num_classes,
            config=factory_config,
        )

    def _save_experiment_metadata(self, exp_dir: Path, spec: ExperimentSpec, config: Dict):
        with open(exp_dir / "experiment_spec.json", "w", encoding="utf-8") as f:
            json.dump(asdict(spec), f, indent=2, ensure_ascii=False)
        with open(exp_dir / "runtime_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    def _load_best_checkpoint(self, model: torch.nn.Module, exp_dir: Path):
        checkpoint_path = exp_dir / "checkpoints" / "best_model.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return checkpoint

    def _save_compatibility_results(self, exp_dir: Path, eval_results: Dict, spec: ExperimentSpec):
        results_dir = exp_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "y_true": eval_results["predictions"]["y_true"],
            "y_pred": eval_results["predictions"]["y_pred"],
            "y_proba": eval_results["predictions"]["y_proba"],
            "metrics": eval_results["basic_metrics"],
            "class_names": self.class_names,
            "confidence_intervals": eval_results.get("confidence_intervals", {}),
            "per_class_metrics": eval_results.get("per_class_metrics", {}),
            "roc_data": eval_results.get("roc_data", {}),
            "pr_data": eval_results.get("pr_data", {}),
            "source_names": [self.source_aliases[idx - 1] for idx in spec.source_indices if idx - 1 < len(self.source_aliases)],
        }
        if eval_results["predictions"].get("attention_weights") is not None:
            payload["attention_weights"] = eval_results["predictions"]["attention_weights"]
        if eval_results["predictions"].get("agentic_actions") is not None:
            payload["agentic_actions"] = eval_results["predictions"]["agentic_actions"]

        with open(results_dir / "test_results.pkl", "wb") as f:
            pickle.dump(payload, f)

    def _generate_experiment_report(
        self,
        exp_dir: Path,
        spec: ExperimentSpec,
        history: Dict,
        eval_results: Dict,
    ) -> str:
        report = ExperimentReport(
            experiment_name=spec.name,
            output_dir=str(exp_dir / "reports"),
            figures_dir=str(exp_dir / "figures"),
        )
        report.add_training_results(history, config=asdict(spec))
        report.add_evaluation_results(
            eval_results["predictions"]["y_true"],
            eval_results["predictions"]["y_pred"],
            eval_results["predictions"]["y_proba"],
            self.class_names,
            eval_results["basic_metrics"],
        )
        attention_weights = eval_results["predictions"].get("attention_weights")
        if attention_weights is not None:
            report.add_attention_analysis(
                attention_weights=attention_weights,
                labels=eval_results["predictions"]["y_true"],
                class_names=self.class_names,
                source_names=[self.source_aliases[idx - 1] for idx in spec.source_indices if idx - 1 < len(self.source_aliases)],
            )
        html_path = report.generate_html_report()
        report.save_report_data()
        return html_path

    def run_experiment(self, spec: ExperimentSpec) -> Dict:
        exp_dir = self.output_dir / spec.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(exp_dir), spec.name)
        logger.info(f"Running formal experiment: {spec.name}")
        logger.info(f"Device: {self.device}")

        source_dims = self._infer_source_dims(spec.source_indices)
        config, factory_config = self._build_config_for_spec(spec, source_dims)
        loaders = self._create_loaders(spec)
        model = self._build_model(spec, factory_config, source_dims)
        self._save_experiment_metadata(exp_dir, spec, config)

        trainer = Trainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=config,
            device=self.device,
            logger=logger,
            output_dir=str(exp_dir),
        )
        history = trainer.train()

        checkpoint = self._load_best_checkpoint(model, exp_dir)
        evaluator = ComprehensiveEvaluator(
            model=model,
            device=self.device,
            class_names=self.class_names,
            output_dir=str(exp_dir),
        )
        eval_results = evaluator.evaluate(loaders["test"])
        evaluator.print_report(eval_results)
        self._save_compatibility_results(exp_dir, eval_results, spec)
        report_path = self._generate_experiment_report(exp_dir, spec, history, eval_results)

        metrics = eval_results["basic_metrics"]
        summary = {
            "name": spec.name,
            "family": spec.family,
            "model_type": spec.model_type,
            "sources": "+".join(self.source_aliases[idx - 1] for idx in spec.source_indices if idx - 1 < len(self.source_aliases)),
            "fusion_type": spec.fusion_type or "",
            "use_agentic": spec.use_agentic,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision_weighted": float(metrics.get("precision_weighted", 0.0)),
            "recall_weighted": float(metrics.get("recall_weighted", 0.0)),
            "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
            "f1_macro": float(metrics.get("f1_macro", 0.0)),
            "auc_roc_macro": float(metrics.get("auc_roc_macro", metrics.get("auc_roc", 0.0))),
            "best_val_acc": float(trainer.best_val_acc),
            "best_val_loss": float(trainer.best_val_loss),
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "num_test_samples": int(len(eval_results["predictions"]["y_true"])),
            "report_path": report_path,
            "experiment_dir": str(exp_dir),
        }
        self.results.append({
            "spec": spec,
            "summary": summary,
            "history": history,
            "eval_results": eval_results,
        })
        self.predictions[spec.name] = eval_results["predictions"]["y_pred"]
        return summary

    def run_many(self, specs: Sequence[ExperimentSpec]) -> pd.DataFrame:
        rows = [self.run_experiment(spec) for spec in specs]
        return pd.DataFrame(rows)

    def _dataset_analysis(self):
        figures_dir = self.output_dir / "dataset_figures"
        report = ExperimentReport(
            experiment_name="bccc_dataset_overview",
            output_dir=str(self.output_dir / "dataset_report"),
            figures_dir=str(figures_dir),
        )
        if self.single_source_data_path and os.path.exists(self.single_source_data_path):
            with open(self.single_source_data_path, "rb") as f:
                single_source = pickle.load(f)
            features = np.vstack([single_source["X_train"], single_source["X_val"], single_source["X_test"]])
            labels = np.concatenate([single_source["y_train"], single_source["y_val"], single_source["y_test"]])
            report.add_data_analysis(features, labels, single_source["feature_names"], single_source["class_names"])
        report.save_report_data()
        report.generate_html_report()

    def _write_tables(self, summary_df: pd.DataFrame):
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(tables_dir / "experiment_summary.csv", index=False)
        summary_df.to_json(tables_dir / "experiment_summary.json", orient="records", force_ascii=False, indent=2)

        latex_df = summary_df[
            ["name", "family", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "auc_roc_macro"]
        ].copy()
        latex_df.columns = [
            "Experiment",
            "Family",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-weighted",
            "F1-macro",
            "AUC",
        ]
        with open(tables_dir / "experiment_summary.tex", "w", encoding="utf-8") as f:
            f.write(latex_df.to_latex(index=False, float_format=lambda value: f"{value:.4f}"))

        per_class_rows = []
        for result in self.results:
            name = result["spec"].name
            per_class = result["eval_results"].get("per_class_metrics", {})
            for class_name, metrics in per_class.items():
                per_class_rows.append({
                    "experiment": name,
                    "class_name": class_name,
                    **metrics,
                })
        per_class_df = pd.DataFrame(per_class_rows)
        if not per_class_df.empty:
            per_class_df.to_csv(tables_dir / "per_class_metrics.csv", index=False)
            with open(tables_dir / "per_class_metrics.tex", "w", encoding="utf-8") as f:
                f.write(per_class_df.to_latex(index=False, float_format=lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)))

        ci_rows = []
        for result in self.results:
            ci = result["eval_results"].get("confidence_intervals", {})
            for metric_name, metric_values in ci.items():
                ci_rows.append({
                    "experiment": result["spec"].name,
                    "metric": metric_name,
                    **metric_values,
                })
        if ci_rows:
            ci_df = pd.DataFrame(ci_rows)
            ci_df.to_csv(tables_dir / "confidence_intervals.csv", index=False)

    def _plot_metric_comparison(self, summary_df: pd.DataFrame):
        fig_dir = self.output_dir / "comparison_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        metric_names = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "auc_roc_macro"]
        melted = summary_df.melt(
            id_vars=["name", "family"],
            value_vars=metric_names,
            var_name="metric",
            value_name="value",
        )
        plt.figure(figsize=(14, 7))
        sns.barplot(data=melted, x="metric", y="value", hue="name")
        plt.ylim(0, 1.05)
        plt.title("Formal Experiment Metric Comparison")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=20)
        plt.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "metric_comparison.png", dpi=200)
        plt.close()

        pivot_rows = []
        for result in self.results:
            for class_name, metrics in result["eval_results"].get("per_class_metrics", {}).items():
                pivot_rows.append({
                    "experiment": result["spec"].name,
                    "class_name": class_name,
                    "f1": metrics.get("f1", 0.0),
                })
        if pivot_rows:
            pivot_df = pd.DataFrame(pivot_rows).pivot(index="experiment", columns="class_name", values="f1")
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".3f")
            plt.title("Per-class F1 Heatmap")
            plt.tight_layout()
            plt.savefig(fig_dir / "per_class_f1_heatmap.png", dpi=200)
            plt.close()

        attention_rows = []
        for result in self.results:
            attention = result["eval_results"]["predictions"].get("attention_weights")
            if attention is None:
                continue
            mean_attention = attention.mean(axis=0)
            source_names = [self.source_aliases[idx - 1] for idx in result["spec"].source_indices if idx - 1 < len(self.source_aliases)]
            for source_name, value in zip(source_names, mean_attention):
                attention_rows.append({
                    "experiment": result["spec"].name,
                    "source": source_name,
                    "attention": float(value),
                })
        if attention_rows:
            attention_df = pd.DataFrame(attention_rows)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=attention_df, x="source", y="attention", hue="experiment")
            plt.title("Mean Source Attention by Experiment")
            plt.tight_layout()
            plt.savefig(fig_dir / "attention_comparison.png", dpi=200)
            plt.close()

        proposed = next((result for result in self.results if result["spec"].name == "flow_log_ti_agentic"), None)
        if proposed is not None:
            actions = proposed["eval_results"]["predictions"].get("agentic_actions")
            if actions is not None and len(actions) > 0:
                action_df = pd.Series(actions).value_counts().rename_axis("action").reset_index(name="count")
                plt.figure(figsize=(10, 5))
                sns.barplot(data=action_df, x="action", y="count", color="#2ecc71")
                plt.title("Agentic Action Distribution")
                plt.xticks(rotation=20)
                plt.tight_layout()
                plt.savefig(fig_dir / "agentic_action_distribution.png", dpi=200)
                plt.close()

        learning_rows = []
        tracked = {"flow_log_attention", "flow_log_ti_decision", "flow_log_ti_agentic"}
        for result in self.results:
            if result["spec"].name not in tracked:
                continue
            for epoch_idx, value in enumerate(result["history"].get("val_f1_macro", []), start=1):
                learning_rows.append({
                    "experiment": result["spec"].name,
                    "epoch": epoch_idx,
                    "val_f1_macro": value,
                })
        if learning_rows:
            learning_df = pd.DataFrame(learning_rows)
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=learning_df, x="epoch", y="val_f1_macro", hue="experiment", marker="o")
            plt.ylim(0, 1.05)
            plt.title("Validation Macro-F1 Across Key Experiments")
            plt.tight_layout()
            plt.savefig(fig_dir / "validation_macro_f1_curves.png", dpi=200)
            plt.close()

    def _write_statistical_tests(self):
        if "flow_log_attention" not in self.predictions or "flow_log_ti_agentic" not in self.predictions:
            return
        y_true = self.results[0]["eval_results"]["predictions"]["y_true"]
        baseline_pred = self.predictions["flow_log_attention"]
        proposed_pred = self.predictions["flow_log_ti_agentic"]
        test = ComprehensiveEvaluator.mcnemar_test(y_true, baseline_pred, proposed_pred)
        stats_dir = self.output_dir / "tables"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "mcnemar_flow_log_attention_vs_agentic.json", "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2, ensure_ascii=False)

    def _write_markdown_report(self, summary_df: pd.DataFrame):
        report_path = self.output_dir / "formal_experiment_report.md"
        best_idx = summary_df["f1_macro"].idxmax()
        best_row = summary_df.loc[best_idx]
        table_text = summary_df.to_string(index=False)
        lines = [
            "# BCCC-CSE-CIC-IDS-2018 Formal Experiment Report",
            "",
            f"- Total experiments: {len(summary_df)}",
            f"- Device: {self.device}",
            f"- Best experiment: {best_row['name']}",
            f"- Best macro F1: {best_row['f1_macro']:.4f}",
            f"- Best accuracy: {best_row['accuracy']:.4f}",
            "",
            "## Experiment Summary",
            "",
            "```text",
            table_text,
            "```",
            "",
            "## Output Files",
            "",
            "- `tables/experiment_summary.csv`",
            "- `tables/experiment_summary.tex`",
            "- `tables/per_class_metrics.csv`",
            "- `comparison_figures/metric_comparison.png`",
            "- `comparison_figures/per_class_f1_heatmap.png`",
            "- `comparison_figures/attention_comparison.png`",
            "- `comparison_figures/validation_macro_f1_curves.png`",
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def finalize(self) -> pd.DataFrame:
        summary_df = pd.DataFrame([item["summary"] for item in self.results])
        self._dataset_analysis()
        self._write_tables(summary_df)
        self._plot_metric_comparison(summary_df)
        self._write_statistical_tests()
        self._write_markdown_report(summary_df)
        return summary_df
