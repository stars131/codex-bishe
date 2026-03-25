"""
Experiment report generation utilities.
"""
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .plots import AttentionVisualizer, DataVisualizer, EvaluationVisualizer, TrainingVisualizer


class ExperimentReport:
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "outputs/reports",
        figures_dir: str = "outputs/figures",
    ):
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(output_dir, experiment_name)
        self.figures_dir = figures_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        self.data_viz = DataVisualizer(os.path.join(figures_dir, "data"))
        self.train_viz = TrainingVisualizer(os.path.join(figures_dir, "training"))
        self.eval_viz = EvaluationVisualizer(os.path.join(figures_dir, "evaluation"))
        self.attn_viz = AttentionVisualizer(os.path.join(figures_dir, "attention"))

        self.report_data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections": {},
        }

    def add_data_analysis(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
    ):
        figures = {}

        fig = self.data_viz.plot_class_distribution(labels, class_names, title="Class Distribution", save_name="class_distribution.png")
        figures["class_distribution"] = fig
        plt.close(fig)

        fig = self.data_viz.plot_correlation_matrix(features, feature_names, top_n=25, save_name="correlation_matrix.png")
        figures["correlation_matrix"] = fig
        plt.close(fig)

        fig = self.data_viz.plot_feature_distribution(features, labels, feature_names, class_names, save_name="feature_distribution.png")
        figures["feature_distribution"] = fig
        plt.close(fig)

        fig = self.data_viz.plot_boxplot_by_class(features, labels, feature_names, class_names, save_name="feature_boxplot.png")
        figures["boxplot"] = fig
        plt.close(fig)

        fig = self.data_viz.plot_data_quality_report(features, feature_names, save_name="data_quality.png")
        figures["data_quality"] = fig
        plt.close(fig)

        try:
            fig = self.data_viz.plot_dimensionality_reduction(features, labels, class_names, method="tsne", n_samples=3000, save_name="tsne_visualization.png")
            figures["tsne"] = fig
            plt.close(fig)
        except Exception as exc:
            print(f"Skip t-SNE visualization: {exc}")

        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            "num_samples": len(labels),
            "num_features": len(feature_names),
            "num_classes": len(class_names),
            "class_distribution": dict(zip([class_names[i] for i in unique], counts.tolist())),
        }

        self.report_data["sections"]["data_analysis"] = {
            "figures": list(figures.keys()),
            "statistics": stats,
        }

    def add_training_results(self, history: Dict[str, List[float]], config: Dict = None):
        figures = {}

        fig = self.train_viz.plot_training_curves(history, title="Training Process", save_name="training_curves.png")
        figures["training_curves"] = fig
        plt.close(fig)

        if "learning_rate" in history:
            fig = self.train_viz.plot_learning_rate(history["learning_rate"], save_name="learning_rate.png")
            figures["learning_rate"] = fig
            plt.close(fig)

        stats = {
            "total_epochs": len(history.get("train_loss", [])),
            "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
            "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
            "best_val_loss": min(history.get("val_loss", [float("inf")])),
            "best_val_acc": max(history.get("val_acc", [0])),
            "config": config,
        }

        self.report_data["sections"]["training"] = {
            "figures": list(figures.keys()),
            "statistics": stats,
        }

    def add_evaluation_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        metrics: Dict[str, float] = None,
    ):
        figures = {}

        fig = self.eval_viz.plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_name="confusion_matrix.png")
        figures["confusion_matrix"] = fig
        plt.close(fig)

        fig = self.eval_viz.plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title="Confusion Matrix (raw)", save_name="confusion_matrix_raw.png")
        figures["confusion_matrix_raw"] = fig
        plt.close(fig)

        try:
            fig = self.eval_viz.plot_roc_curves(y_true, y_proba, class_names, save_name="roc_curves.png")
            figures["roc_curves"] = fig
            plt.close(fig)
        except Exception as exc:
            print(f"Skip ROC curves: {exc}")

        try:
            fig = self.eval_viz.plot_precision_recall_curves(y_true, y_proba, class_names, save_name="pr_curves.png")
            figures["pr_curves"] = fig
            plt.close(fig)
        except Exception as exc:
            print(f"Skip PR curves: {exc}")

        fig = self.eval_viz.plot_per_class_metrics(y_true, y_pred, class_names, save_name="per_class_metrics.png")
        figures["per_class_metrics"] = fig
        plt.close(fig)

        self.report_data["sections"]["evaluation"] = {
            "figures": list(figures.keys()),
            "metrics": metrics,
        }

    def add_attention_analysis(
        self,
        attention_weights: np.ndarray,
        labels: np.ndarray = None,
        class_names: List[str] = None,
        source_names: List[str] = None,
    ):
        source_names = source_names or [f"Source {i + 1}" for i in range(attention_weights.shape[1])]
        figures = {}

        fig = self.attn_viz.plot_attention_weights(attention_weights, source_names, save_name="attention_weights.png")
        figures["attention_weights"] = fig
        plt.close(fig)

        if labels is not None and class_names is not None:
            fig = self.attn_viz.plot_attention_by_class(attention_weights, labels, class_names, source_names, save_name="attention_by_class.png")
            figures["attention_by_class"] = fig
            plt.close(fig)

        self.report_data["sections"]["attention"] = {
            "figures": list(figures.keys()),
            "statistics": {
                "mean_weights": np.mean(attention_weights, axis=0).tolist(),
                "std_weights": np.std(attention_weights, axis=0).tolist(),
                "source_names": source_names,
            },
        }

    def add_model_comparison(self, comparison_results: Dict[str, Dict[str, float]], metric_names: List[str] = None):
        metric_names = metric_names or ["accuracy", "precision", "recall", "f1_score"]
        fig = self.eval_viz.plot_metrics_comparison(comparison_results, metric_names, save_name="model_comparison.png")
        plt.close(fig)
        self.report_data["sections"]["comparison"] = {
            "figures": ["model_comparison"],
            "results": comparison_results,
        }

    def generate_html_report(self) -> str:
        def _fmt(val, fmt: str = ".4f") -> str:
            try:
                return f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                return str(val) if val is not None else "N/A"

        try:
            figures_rel_path = os.path.relpath(self.figures_dir, self.output_dir)
        except ValueError:
            # Windows raises on cross-drive relative paths; absolute path still works in HTML.
            figures_rel_path = os.path.abspath(self.figures_dir)
        html_sections = []

        if "data_analysis" in self.report_data["sections"]:
            stats = self.report_data["sections"]["data_analysis"]["statistics"]
            html_sections.append(f"""
    <div class="section">
        <h2>Data Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card"><div class="value">{stats.get('num_samples', 'N/A'):,}</div><div class="label">Samples</div></div>
            <div class="metric-card"><div class="value">{stats.get('num_features', 'N/A')}</div><div class="label">Features</div></div>
            <div class="metric-card"><div class="value">{stats.get('num_classes', 'N/A')}</div><div class="label">Classes</div></div>
        </div>
        <div class="figure-container"><img src="{figures_rel_path}/data/class_distribution.png" alt="class distribution"></div>
        <div class="figure-container"><img src="{figures_rel_path}/data/correlation_matrix.png" alt="correlation"></div>
        <div class="figure-container"><img src="{figures_rel_path}/data/data_quality.png" alt="data quality"></div>
    </div>
""")

        if "training" in self.report_data["sections"]:
            stats = self.report_data["sections"]["training"]["statistics"]
            html_sections.append(f"""
    <div class="section">
        <h2>Training</h2>
        <div class="metrics-grid">
            <div class="metric-card"><div class="value">{stats.get('total_epochs', 'N/A')}</div><div class="label">Epochs</div></div>
            <div class="metric-card"><div class="value">{_fmt(stats.get('best_val_loss'))}</div><div class="label">Best Val Loss</div></div>
            <div class="metric-card"><div class="value">{_fmt(stats.get('best_val_acc'))}</div><div class="label">Best Val Acc</div></div>
        </div>
        <div class="figure-container"><img src="{figures_rel_path}/training/training_curves.png" alt="training curves"></div>
    </div>
""")

        if "evaluation" in self.report_data["sections"]:
            metrics = self.report_data["sections"]["evaluation"].get("metrics", {}) or {}
            html_sections.append(f"""
    <div class="section">
        <h2>Evaluation</h2>
        <div class="metrics-grid">
            <div class="metric-card"><div class="value">{_fmt(metrics.get('accuracy', 0))}</div><div class="label">Accuracy</div></div>
            <div class="metric-card"><div class="value">{_fmt(metrics.get('precision', metrics.get('precision_weighted', 0)))}</div><div class="label">Precision</div></div>
            <div class="metric-card"><div class="value">{_fmt(metrics.get('recall', metrics.get('recall_weighted', 0)))}</div><div class="label">Recall</div></div>
            <div class="metric-card"><div class="value">{_fmt(metrics.get('f1_score', metrics.get('f1_weighted', 0)))}</div><div class="label">F1</div></div>
        </div>
        <div class="figure-container"><img src="{figures_rel_path}/evaluation/confusion_matrix.png" alt="confusion matrix"></div>
        <div class="figure-container"><img src="{figures_rel_path}/evaluation/per_class_metrics.png" alt="per class metrics"></div>
    </div>
""")

        if "attention" in self.report_data["sections"]:
            stats = self.report_data["sections"]["attention"]["statistics"]
            html_sections.append(f"""
    <div class="section">
        <h2>Attention</h2>
        <p>Sources: {', '.join(stats.get('source_names', []))}</p>
        <p>Mean Weights: {', '.join(f'{w:.3f}' for w in stats.get('mean_weights', []))}</p>
        <div class="figure-container"><img src="{figures_rel_path}/attention/attention_weights.png" alt="attention weights"></div>
    </div>
""")

        if "comparison" in self.report_data["sections"]:
            html_sections.append(f"""
    <div class="section">
        <h2>Comparison</h2>
        <div class="figure-container"><img src="{figures_rel_path}/evaluation/model_comparison.png" alt="model comparison"></div>
    </div>
""")

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {self.experiment_name}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-card .label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .figure-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Network Attack Detection Report</h1>
        <p>Experiment: {self.experiment_name}</p>
        <p>Generated: {self.report_data['timestamp']}</p>
    </div>
    {''.join(html_sections)}
</body>
</html>
"""

        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return html_path

    def save_report_data(self):
        data_path = os.path.join(self.output_dir, "report_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(self.report_data, f)


def generate_full_report(
    experiment_name: str,
    data_path: str = None,
    results_path: str = None,
    history_path: str = None,
    output_dir: str = "outputs/reports",
) -> str:
    report = ExperimentReport(experiment_name, output_dir)

    if data_path and os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        all_features = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
        all_labels = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
        report.add_data_analysis(all_features, all_labels, data["feature_names"], data["class_names"])

    if history_path and os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history_data = pickle.load(f)
        report.add_training_results(history_data.get("history", history_data))

    if results_path and os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        report.add_evaluation_results(
            results["y_true"],
            results["y_pred"],
            results["y_proba"],
            results["class_names"],
            results.get("metrics"),
        )

        if results.get("attention_weights") is not None:
            report.add_attention_analysis(
                results["attention_weights"],
                results.get("y_true"),
                results.get("class_names"),
                results.get("source_names"),
            )

    html_path = report.generate_html_report()
    report.save_report_data()
    return html_path
