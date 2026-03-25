"""
Core visualization utilities.
"""
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.figsize"] = (12, 8)

COLORS = {
    "primary": "#3498db",
    "secondary": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#9b59b6",
    "dark": "#34495e",
    "light": "#ecf0f1",
}

COLOR_PALETTE = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
    "#27ae60", "#8e44ad", "#2980b9", "#d35400", "#7f8c8d",
]


class PlotStyle:
    @staticmethod
    def set_chinese_font():
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    @staticmethod
    def set_publication_style():
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
        })

    @staticmethod
    def get_color_palette(n_colors: int) -> List[str]:
        if n_colors <= len(COLOR_PALETTE):
            return COLOR_PALETTE[:n_colors]
        return plt.cm.tab20(np.linspace(0, 1, n_colors)).tolist()


class _BaseVisualizer:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        PlotStyle.set_chinese_font()

    def _save_figure(self, fig: plt.Figure, save_name: Optional[str]):
        if save_name:
            path = os.path.join(self.save_dir, save_name)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"Figure saved: {path}")


class DataVisualizer(_BaseVisualizer):
    def __init__(self, save_dir: str = "outputs/figures/data"):
        super().__init__(save_dir)

    def plot_class_distribution(
        self,
        labels: np.ndarray,
        class_names: List[str],
        title: str = "Class Distribution",
        figsize: Tuple[int, int] = (14, 5),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        names = [class_names[i] for i in unique]
        colors = PlotStyle.get_color_palette(len(unique))

        bars = axes[0].bar(range(len(unique)), counts, color=colors, edgecolor="white", linewidth=1.5)
        axes[0].set_xticks(range(len(unique)))
        axes[0].set_xticklabels(names, rotation=45, ha="right")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"{title} - Bar")

        for bar, count, pct in zip(bars, counts, percentages):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

        axes[1].pie(counts, labels=names, autopct="%1.1f%%", colors=colors, explode=[0.02] * len(unique), shadow=True, startangle=90)
        axes[1].set_title(f"{title} - Pie")

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        title: str = "Feature Importance",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        indices = np.argsort(importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_importance, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Importance score")
        ax.set_title(title)

        for bar, val in zip(bars, top_importance):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_correlation_matrix(
        self,
        features: np.ndarray,
        feature_names: List[str],
        top_n: int = 30,
        method: str = "pearson",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        if len(feature_names) > top_n:
            variances = np.var(features, axis=0)
            top_indices = np.argsort(variances)[-top_n:]
            features = features[:, top_indices]
            feature_names = [feature_names[i] for i in top_indices]

        df = pd.DataFrame(features, columns=feature_names)
        corr_matrix = df.corr(method=method)
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0, annot=False, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Correlation"}, ax=ax)
        ax.set_title(f"Feature Correlation (Top {top_n})")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_feature_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        feature_indices: List[int] = None,
        n_cols: int = 2,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        if feature_indices is None:
            variances = np.var(features, axis=0)
            feature_indices = np.argsort(variances)[-4:]

        n_features = len(feature_indices)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        colors = PlotStyle.get_color_palette(len(np.unique(labels)))

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            feat_name = feature_names[feat_idx]
            for j, cls_idx in enumerate(np.unique(labels)):
                cls_data = features[labels == cls_idx, feat_idx]
                ax.hist(cls_data, bins=50, alpha=0.6, label=class_names[cls_idx], color=colors[j], density=True)
            ax.set_xlabel(feat_name)
            ax.set_ylabel("Density")
            ax.set_title(f"{feat_name} distribution")
            ax.legend(fontsize=8)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_dimensionality_reduction(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        method: str = "tsne",
        n_samples: int = 5000,
        perplexity: int = 30,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        if len(features) > n_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(features), n_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity) if method == "tsne" else PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(features)

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = PlotStyle.get_color_palette(len(np.unique(labels)))
        for i, cls_idx in enumerate(np.unique(labels)):
            mask = labels == cls_idx
            ax.scatter(reduced[mask, 0], reduced[mask, 1], c=[colors[i]], label=class_names[cls_idx], alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.set_title(f"{method.upper()} visualization")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_boxplot_by_class(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        feature_indices: List[int] = None,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        if feature_indices is None:
            variances = np.var(features, axis=0)
            feature_indices = np.argsort(variances)[-6:]

        n_features = len(feature_indices)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        colors = PlotStyle.get_color_palette(len(np.unique(labels)))

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            feat_name = feature_names[feat_idx]
            data = [features[labels == cls_idx, feat_idx] for cls_idx in np.unique(labels)]
            bp = ax.boxplot(data, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels([class_names[idx] for idx in np.unique(labels)], rotation=45, ha="right")
            ax.set_title(feat_name)
            ax.grid(True, alpha=0.3)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_data_quality_report(
        self,
        features: np.ndarray,
        feature_names: List[str],
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        missing = np.isnan(features).sum(axis=0)
        missing_pct = missing / len(features) * 100
        top_missing_idx = np.argsort(missing_pct)[-10:]
        ax1.barh(range(len(top_missing_idx)), missing_pct[top_missing_idx], color=COLORS["warning"])
        ax1.set_yticks(range(len(top_missing_idx)))
        ax1.set_yticklabels([feature_names[i][:20] for i in top_missing_idx])
        ax1.set_xlabel("Missing %")
        ax1.set_title("Top missing features")

        ax2 = fig.add_subplot(gs[0, 1])
        variances = np.var(features, axis=0)
        ax2.hist(variances, bins=50, color=COLORS["primary"], edgecolor="white")
        ax2.set_xlabel("Variance")
        ax2.set_ylabel("Count")
        ax2.set_title("Variance distribution")
        ax2.axvline(np.median(variances), color="red", linestyle="--", label=f"median={np.median(variances):.2f}")
        ax2.legend()

        ax3 = fig.add_subplot(gs[0, 2])
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1
        outliers_total = np.sum(features < (q1 - 1.5 * iqr), axis=0) + np.sum(features > (q3 + 1.5 * iqr), axis=0)
        outlier_pct = outliers_total / len(features) * 100
        top_outlier_idx = np.argsort(outlier_pct)[-10:]
        ax3.barh(range(len(top_outlier_idx)), outlier_pct[top_outlier_idx], color=COLORS["danger"])
        ax3.set_yticks(range(len(top_outlier_idx)))
        ax3.set_yticklabels([feature_names[i][:20] for i in top_outlier_idx])
        ax3.set_xlabel("Outlier %")
        ax3.set_title("Top outlier features")

        ax4 = fig.add_subplot(gs[1, 0])
        ranges = np.max(features, axis=0) - np.min(features, axis=0)
        ax4.hist(np.log10(ranges + 1e-10), bins=50, color=COLORS["secondary"], edgecolor="white")
        ax4.set_xlabel("log10(range)")
        ax4.set_ylabel("Count")
        ax4.set_title("Feature range distribution")

        ax5 = fig.add_subplot(gs[1, 1])
        sample_means = np.mean(features, axis=1)
        ax5.hist(sample_means, bins=50, color=COLORS["info"], edgecolor="white")
        ax5.set_xlabel("Sample mean")
        ax5.set_ylabel("Count")
        ax5.set_title("Sample mean distribution")

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        stats_text = (
            f"Samples: {features.shape[0]:,}\n"
            f"Features: {features.shape[1]}\n\n"
            f"Missing values: {np.isnan(features).sum():,}\n"
            f"Missing ratio: {np.isnan(features).sum() / features.size * 100:.2f}%\n"
            f"Zero ratio: {(features == 0).sum() / features.size * 100:.2f}%\n\n"
            f"Variance min: {variances.min():.4f}\n"
            f"Variance max: {variances.max():.4f}\n"
            f"Variance mean: {variances.mean():.4f}"
        )
        ax6.text(0.1, 0.5, stats_text, fontsize=12, family="monospace", verticalalignment="center", transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.suptitle("Data Quality Report", fontsize=16, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig


class TrainingVisualizer(_BaseVisualizer):
    def __init__(self, save_dir: str = "outputs/figures/training"):
        super().__init__(save_dir)

    def plot_training_curves(self, history: Dict[str, List[float]], title: str = "Training Curves", save_name: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        if "train_loss" in history:
            axes[0].plot(epochs, history["train_loss"], "b-", linewidth=2, label="train", marker="o", markersize=3)
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], "r-", linewidth=2, label="val", marker="s", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "val_loss" in history and history["val_loss"]:
            best_epoch = np.argmin(history["val_loss"]) + 1
            best_loss = min(history["val_loss"])
            axes[0].axvline(best_epoch, color="green", linestyle="--", alpha=0.7)
            axes[0].annotate(f"Best: {best_loss:.4f}", xy=(best_epoch, best_loss), xytext=(best_epoch + 1, best_loss + 0.05), fontsize=10, color="green")

        if "train_acc" in history:
            axes[1].plot(epochs, history["train_acc"], "b-", linewidth=2, label="train", marker="o", markersize=3)
        if "val_acc" in history:
            axes[1].plot(epochs, history["val_acc"], "r-", linewidth=2, label="val", marker="s", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_learning_rate(self, lr_history: List[float], save_name: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, "g-", linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning rate")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_metrics_over_time(self, metrics_history: Dict[str, List[float]], metric_names: List[str] = None, save_name: Optional[str] = None) -> plt.Figure:
        metric_names = metric_names or list(metrics_history.keys())
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        colors = PlotStyle.get_color_palette(n_metrics)

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            if metric in metrics_history:
                epochs = range(1, len(metrics_history[metric]) + 1)
                ax.plot(epochs, metrics_history[metric], color=colors[i], linewidth=2, marker="o", markersize=3)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.set_title(metric)
                ax.grid(True, alpha=0.3)

        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_gradient_flow(self, named_parameters, save_name: Optional[str] = None) -> plt.Figure:
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().numpy())
                max_grads.append(p.grad.abs().max().cpu().numpy())

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max")
        ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean")
        ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Gradient")
        ax.set_title("Gradient flow")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig


class EvaluationVisualizer(_BaseVisualizer):
    def __init__(self, save_dir: str = "outputs/figures/evaluation"):
        super().__init__(save_dir)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], normalize: bool = True, title: str = "Confusion Matrix", save_name: Optional[str] = None) -> plt.Figure:
        cm = confusion_matrix(y_true, y_pred)
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
        fmt = ".2%" if normalize else "d"

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 10})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], title: str = "ROC Curves", save_name: Optional[str] = None) -> plt.Figure:
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = PlotStyle.get_color_palette(n_classes)

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if n_classes == 2 and i == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i] if n_classes > 2 else y_true, y_proba[:, i] if n_classes > 2 else y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{class_name} (AUC={roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="random")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_precision_recall_curves(self, y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], title: str = "PR Curves", save_name: Optional[str] = None) -> plt.Figure:
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = PlotStyle.get_color_palette(n_classes)

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if n_classes == 2 and i == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i] if n_classes > 2 else y_true, y_proba[:, i] if n_classes > 2 else y_proba[:, 1])
            ap = average_precision_score(y_true_bin[:, i] if n_classes > 2 else y_true, y_proba[:, i] if n_classes > 2 else y_proba[:, 1])
            ax.plot(recall, precision, color=color, linewidth=2, label=f"{class_name} (AP={ap:.4f})")

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]], metric_names: List[str] = None, title: str = "Model Comparison", save_name: Optional[str] = None) -> plt.Figure:
        metric_names = metric_names or ["accuracy", "precision", "recall", "f1_score"]
        model_names = list(metrics_dict.keys())
        n_models = len(model_names)
        x = np.arange(len(metric_names))
        width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = PlotStyle.get_color_palette(n_models)

        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name].get(metric, 0) for metric in metric_names]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i], edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], title: str = "Per-class Metrics", save_name: Optional[str] = None) -> plt.Figure:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        metrics = ["precision", "recall", "f1-score"]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(class_names))
        width = 0.25
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["warning"]]

        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in class_names]
            ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i], edgecolor="white")

        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig


class AttentionVisualizer(_BaseVisualizer):
    def __init__(self, save_dir: str = "outputs/figures/attention"):
        super().__init__(save_dir)

    def plot_attention_weights(self, attention_weights: np.ndarray, source_names: List[str] = None, title: str = "Attention Weights", save_name: Optional[str] = None) -> plt.Figure:
        attention_weights = np.asarray(attention_weights)
        if attention_weights.ndim != 2:
            raise ValueError(f"attention_weights should be 2D, got shape {attention_weights.shape}")

        if source_names is None or len(source_names) != attention_weights.shape[1]:
            source_names = [f"Source {i + 1}" for i in range(attention_weights.shape[1])]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = PlotStyle.get_color_palette(len(source_names))

        mean_weights = np.mean(attention_weights, axis=0)
        std_weights = np.std(attention_weights, axis=0)
        bars = axes[0].bar(source_names, mean_weights, yerr=std_weights, color=colors, edgecolor="white", capsize=5)
        axes[0].set_ylabel("Mean attention weight")
        axes[0].set_title("Average attention by source")
        axes[0].tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, mean_weights):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=10)
        axes[0].set_ylim(0, 1)

        for i, (name, color) in enumerate(zip(source_names, colors)):
            axes[1].hist(attention_weights[:, i], bins=50, alpha=0.6, label=name, color=color, edgecolor="white")
        axes[1].set_xlabel("Attention weight")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Weight distribution")
        axes[1].legend(fontsize=9)

        if attention_weights.shape[1] >= 2:
            axes[2].scatter(attention_weights[:, 0], attention_weights[:, 1], alpha=0.3, s=10, c=colors[0])
            axes[2].plot([0, 1], [1, 0], "r--", linewidth=2, label="sum=1 guide")
            axes[2].set_xlabel(f"{source_names[0]} weight")
            axes[2].set_ylabel(f"{source_names[1]} weight")
            axes[2].set_title("First two sources")
            axes[2].legend(fontsize=9)
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
        else:
            axes[2].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_attention_by_class(self, attention_weights: np.ndarray, labels: np.ndarray, class_names: List[str], source_names: List[str] = None, title: str = "Attention By Class", save_name: Optional[str] = None) -> plt.Figure:
        attention_weights = np.asarray(attention_weights)
        labels = np.asarray(labels)

        if source_names is None or len(source_names) != attention_weights.shape[1]:
            source_names = [f"Source {i + 1}" for i in range(attention_weights.shape[1])]

        n_classes = len(class_names)
        n_sources = len(source_names)
        fig, ax = plt.subplots(figsize=(max(12, n_sources * 2.4), 6))

        x = np.arange(n_classes)
        width = 0.8 / max(n_sources, 1)
        colors = PlotStyle.get_color_palette(n_sources)

        for i, (source, color) in enumerate(zip(source_names, colors)):
            mean_weights = []
            std_weights = []
            for cls_idx in range(n_classes):
                cls_mask = labels == cls_idx
                if np.any(cls_mask):
                    mean_weights.append(np.mean(attention_weights[cls_mask, i]))
                    std_weights.append(np.std(attention_weights[cls_mask, i]))
                else:
                    mean_weights.append(0.0)
                    std_weights.append(0.0)

            offset = (i - n_sources / 2 + 0.5) * width
            ax.bar(x + offset, mean_weights, width, yerr=std_weights, label=source, color=color, edgecolor="white", capsize=3)

        ax.set_xlabel("Class")
        ax.set_ylabel("Mean attention weight")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend(fontsize=9, ncol=min(3, n_sources))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_attention_heatmap(self, attention_matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title: str = "Attention Heatmap", save_name: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_matrix, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=col_labels, yticklabels=row_labels, ax=ax, cbar_kws={"label": "Attention weight"})
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
