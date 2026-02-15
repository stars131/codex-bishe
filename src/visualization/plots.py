"""
可视化核心模块

提供完整的数据分析、训练监控和模型评估可视化功能。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格（兼容不同版本的matplotlib）
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')  # 通用fallback
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 8)

# 自定义颜色方案
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#9b59b6',
    'dark': '#34495e',
    'light': '#ecf0f1'
}

COLOR_PALETTE = [
    '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
    '#27ae60', '#8e44ad', '#2980b9', '#d35400', '#7f8c8d'
]


class PlotStyle:
    """绘图样式管理器"""

    @staticmethod
    def set_chinese_font():
        """设置中文字体"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    def set_publication_style():
        """设置论文发表风格"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18
        })

    @staticmethod
    def get_color_palette(n_colors: int) -> List[str]:
        """获取颜色调色板"""
        if n_colors <= len(COLOR_PALETTE):
            return COLOR_PALETTE[:n_colors]
        return plt.cm.tab20(np.linspace(0, 1, n_colors)).tolist()


class DataVisualizer:
    """
    数据可视化器

    用于数据探索、特征分析和数据质量评估。
    """

    def __init__(self, save_dir: str = "outputs/figures/data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        PlotStyle.set_chinese_font()

    def plot_class_distribution(
        self,
        labels: np.ndarray,
        class_names: List[str],
        title: str = "类别分布",
        figsize: Tuple[int, int] = (14, 5),
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制类别分布图（柱状图+饼图）"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        colors = PlotStyle.get_color_palette(len(unique))

        # 柱状图
        bars = axes[0].bar(range(len(unique)), counts, color=colors, edgecolor='white', linewidth=1.5)
        axes[0].set_xticks(range(len(unique)))
        axes[0].set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
        axes[0].set_ylabel('样本数量', fontsize=12)
        axes[0].set_title(f'{title} - 柱状图', fontsize=14, fontweight='bold')

        for bar, count, pct in zip(bars, counts, percentages):
            axes[0].text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9
            )

        # 饼图
        wedges, texts, autotexts = axes[1].pie(
            counts,
            labels=[class_names[i] for i in unique],
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(unique),
            shadow=True,
            startangle=90
        )
        axes[1].set_title(f'{title} - 饼图', fontsize=14, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        title: str = "特征重要性",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制特征重要性图"""
        # 排序
        indices = np.argsort(importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_importance, color=colors)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        for bar, val in zip(bars, top_importance):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_correlation_matrix(
        self,
        features: np.ndarray,
        feature_names: List[str],
        top_n: int = 30,
        method: str = 'pearson',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制特征相关性矩阵热力图"""
        # 选择方差最大的特征
        if len(feature_names) > top_n:
            variances = np.var(features, axis=0)
            top_indices = np.argsort(variances)[-top_n:]
            features = features[:, top_indices]
            feature_names = [feature_names[i] for i in top_indices]

        # 计算相关性
        df = pd.DataFrame(features, columns=feature_names)
        corr_matrix = df.corr(method=method)

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            annot=False, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "相关系数"},
            ax=ax
        )

        ax.set_title(f'特征相关性矩阵 (Top {top_n})', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
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
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制各类别的特征分布"""
        if feature_indices is None:
            variances = np.var(features, axis=0)
            feature_indices = np.argsort(variances)[-4:]

        n_features = len(feature_indices)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        colors = PlotStyle.get_color_palette(len(np.unique(labels)))

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            feat_name = feature_names[feat_idx]

            for j, cls_idx in enumerate(np.unique(labels)):
                cls_data = features[labels == cls_idx, feat_idx]
                ax.hist(cls_data, bins=50, alpha=0.6, label=class_names[cls_idx],
                       color=colors[j], density=True)

            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel('密度', fontsize=10)
            ax.set_title(f'{feat_name} 分布', fontsize=12)
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
        method: str = 'tsne',
        n_samples: int = 5000,
        perplexity: int = 30,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """降维可视化（t-SNE/PCA）"""
        # 采样
        if len(features) > n_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(features), n_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        print(f"执行 {method.upper()} 降维...")

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"不支持的方法: {method}")

        reduced = reducer.fit_transform(features)

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = PlotStyle.get_color_palette(len(np.unique(labels)))

        for i, cls_idx in enumerate(np.unique(labels)):
            mask = labels == cls_idx
            ax.scatter(reduced[mask, 0], reduced[mask, 1], c=[colors[i]],
                      label=class_names[cls_idx], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

        ax.set_xlabel(f'{method.upper()} 维度 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} 维度 2', fontsize=12)
        ax.set_title(f'{method.upper()} 可视化', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
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
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制各类别的特征箱线图"""
        if feature_indices is None:
            variances = np.var(features, axis=0)
            feature_indices = np.argsort(variances)[-6:]

        n_features = len(feature_indices)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        colors = PlotStyle.get_color_palette(len(np.unique(labels)))

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            feat_name = feature_names[feat_idx]
            data = [features[labels == cls_idx, feat_idx] for cls_idx in np.unique(labels)]

            bp = ax.boxplot(data, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticklabels([class_names[idx] for idx in np.unique(labels)], rotation=45, ha='right')
            ax.set_title(feat_name, fontsize=11)
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
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制数据质量报告"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # 1. 缺失值统计
        ax1 = fig.add_subplot(gs[0, 0])
        missing = np.isnan(features).sum(axis=0)
        missing_pct = missing / len(features) * 100
        top_missing_idx = np.argsort(missing_pct)[-10:]

        ax1.barh(range(len(top_missing_idx)), missing_pct[top_missing_idx], color=COLORS['warning'])
        ax1.set_yticks(range(len(top_missing_idx)))
        ax1.set_yticklabels([feature_names[i][:20] for i in top_missing_idx])
        ax1.set_xlabel('缺失率 (%)')
        ax1.set_title('缺失值统计 (Top 10)', fontweight='bold')

        # 2. 特征方差分布
        ax2 = fig.add_subplot(gs[0, 1])
        variances = np.var(features, axis=0)
        ax2.hist(variances, bins=50, color=COLORS['primary'], edgecolor='white')
        ax2.set_xlabel('方差')
        ax2.set_ylabel('频数')
        ax2.set_title('特征方差分布', fontweight='bold')
        ax2.axvline(np.median(variances), color='red', linestyle='--', label=f'中位数: {np.median(variances):.2f}')
        ax2.legend()

        # 3. 异常值统计
        ax3 = fig.add_subplot(gs[0, 2])
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1
        outliers_low = np.sum(features < (q1 - 1.5 * iqr), axis=0)
        outliers_high = np.sum(features > (q3 + 1.5 * iqr), axis=0)
        outliers_total = outliers_low + outliers_high
        outlier_pct = outliers_total / len(features) * 100

        top_outlier_idx = np.argsort(outlier_pct)[-10:]
        ax3.barh(range(len(top_outlier_idx)), outlier_pct[top_outlier_idx], color=COLORS['danger'])
        ax3.set_yticks(range(len(top_outlier_idx)))
        ax3.set_yticklabels([feature_names[i][:20] for i in top_outlier_idx])
        ax3.set_xlabel('异常值比例 (%)')
        ax3.set_title('异常值统计 (Top 10)', fontweight='bold')

        # 4. 特征值范围
        ax4 = fig.add_subplot(gs[1, 0])
        ranges = np.max(features, axis=0) - np.min(features, axis=0)
        ax4.hist(np.log10(ranges + 1e-10), bins=50, color=COLORS['secondary'], edgecolor='white')
        ax4.set_xlabel('log10(特征范围)')
        ax4.set_ylabel('频数')
        ax4.set_title('特征值范围分布', fontweight='bold')

        # 5. 样本统计
        ax5 = fig.add_subplot(gs[1, 1])
        sample_means = np.mean(features, axis=1)
        ax5.hist(sample_means, bins=50, color=COLORS['info'], edgecolor='white')
        ax5.set_xlabel('样本均值')
        ax5.set_ylabel('频数')
        ax5.set_title('样本均值分布', fontweight='bold')

        # 6. 总体统计信息
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        stats_text = f"""
        数据集统计信息
        ─────────────────
        样本数量: {features.shape[0]:,}
        特征数量: {features.shape[1]}

        缺失值总数: {np.isnan(features).sum():,}
        缺失值比例: {np.isnan(features).sum() / features.size * 100:.2f}%

        零值比例: {(features == 0).sum() / features.size * 100:.2f}%

        特征方差:
          最小: {variances.min():.4f}
          最大: {variances.max():.4f}
          平均: {variances.mean():.4f}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('数据质量报告', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def _save_figure(self, fig: plt.Figure, save_name: Optional[str]):
        """保存图片"""
        if save_name:
            path = os.path.join(self.save_dir, save_name)
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f"图片已保存: {path}")


class TrainingVisualizer:
    """
    训练过程可视化器

    用于监控和分析模型训练过程。
    """

    def __init__(self, save_dir: str = "outputs/figures/training"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        PlotStyle.set_chinese_font()

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "训练曲线",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制训练曲线（损失+准确率）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history.get('train_loss', [])) + 1)

        # Loss曲线
        if 'train_loss' in history:
            axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='训练损失', marker='o', markersize=3)
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='验证损失', marker='s', markersize=3)

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('损失曲线', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 标记最佳点
        if 'val_loss' in history:
            best_epoch = np.argmin(history['val_loss']) + 1
            best_loss = min(history['val_loss'])
            axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.7)
            axes[0].annotate(f'Best: {best_loss:.4f}', xy=(best_epoch, best_loss),
                           xytext=(best_epoch+2, best_loss+0.05),
                           fontsize=10, color='green')

        # Accuracy曲线
        if 'train_acc' in history:
            axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='训练准确率', marker='o', markersize=3)
        if 'val_acc' in history:
            axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='验证准确率', marker='s', markersize=3)

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('准确率曲线', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_learning_rate(
        self,
        lr_history: List[float],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制学习率变化曲线"""
        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('学习率变化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_metrics_over_time(
        self,
        metrics_history: Dict[str, List[float]],
        metric_names: List[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制多个指标随时间变化"""
        if metric_names is None:
            metric_names = list(metrics_history.keys())

        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        colors = PlotStyle.get_color_palette(n_metrics)

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            if metric in metrics_history:
                epochs = range(1, len(metrics_history[metric]) + 1)
                ax.plot(epochs, metrics_history[metric], color=colors[i], linewidth=2, marker='o', markersize=3)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(metric, fontweight='bold')
                ax.grid(True, alpha=0.3)

        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_gradient_flow(
        self,
        named_parameters,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制梯度流"""
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().numpy())
                max_grads.append(p.grad.abs().max().cpu().numpy())

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color='c', label='最大梯度')
        ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color='b', label='平均梯度')
        ax.hlines(0, 0, len(ave_grads)+1, lw=2, color='k')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=8)
        ax.set_xlabel('网络层')
        ax.set_ylabel('梯度值')
        ax.set_title('梯度流', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def _save_figure(self, fig: plt.Figure, save_name: Optional[str]):
        if save_name:
            path = os.path.join(self.save_dir, save_name)
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f"图片已保存: {path}")


class EvaluationVisualizer:
    """
    模型评估可视化器

    用于展示模型性能评估结果。
    """

    def __init__(self, save_dir: str = "outputs/figures/evaluation"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        PlotStyle.set_chinese_font()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        title: str = "混淆矩阵",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            cm_display = cm
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm_display, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 10}
        )

        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        title: str = "ROC 曲线",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制多类别ROC曲线"""
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = PlotStyle.get_color_palette(n_classes)

        # 计算每个类的ROC曲线
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if n_classes == 2 and i == 0:
                continue  # 二分类只画正类

            fpr, tpr, _ = roc_curve(y_true_bin[:, i] if n_classes > 2 else y_true,
                                    y_proba[:, i] if n_classes > 2 else y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{class_name} (AUC = {roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机猜测')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
        ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        title: str = "PR 曲线",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制精确率-召回率曲线"""
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = PlotStyle.get_color_palette(n_classes)

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if n_classes == 2 and i == 0:
                continue

            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i] if n_classes > 2 else y_true,
                y_proba[:, i] if n_classes > 2 else y_proba[:, 1]
            )
            ap = average_precision_score(
                y_true_bin[:, i] if n_classes > 2 else y_true,
                y_proba[:, i] if n_classes > 2 else y_proba[:, 1]
            )

            ax.plot(recall, precision, color=color, linewidth=2,
                   label=f'{class_name} (AP = {ap:.4f})')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('召回率 (Recall)', fontsize=12)
        ax.set_ylabel('精确率 (Precision)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_names: List[str] = None,
        title: str = "模型性能对比",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制不同模型的性能对比图"""
        if metric_names is None:
            metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        model_names = list(metrics_dict.keys())
        n_models = len(model_names)
        n_metrics = len(metric_names)

        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = PlotStyle.get_color_palette(n_models)

        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name].get(m, 0) for m in metric_names]
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i], edgecolor='white')

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)

        ax.set_xlabel('评估指标', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        title: str = "各类别性能指标",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制每个类别的详细指标"""
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        metrics = ['precision', 'recall', 'f1-score']
        n_classes = len(class_names)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_classes)
        width = 0.25
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['warning']]

        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in class_names]
            ax.bar(x + i*width, values, width, label=metric.capitalize(), color=colors[i], edgecolor='white')

        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def _save_figure(self, fig: plt.Figure, save_name: Optional[str]):
        if save_name:
            path = os.path.join(self.save_dir, save_name)
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f"图片已保存: {path}")


class AttentionVisualizer:
    """
    注意力机制可视化器

    用于分析和展示注意力权重。
    """

    def __init__(self, save_dir: str = "outputs/figures/attention"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        PlotStyle.set_chinese_font()

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        source_names: List[str] = None,
        title: str = "注意力权重分析",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制注意力权重分布"""
        if source_names is None:
            source_names = [f'数据源 {i+1}' for i in range(attention_weights.shape[1])]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        colors = [COLORS['primary'], COLORS['danger']]

        # 1. 平均权重柱状图
        mean_weights = np.mean(attention_weights, axis=0)
        std_weights = np.std(attention_weights, axis=0)

        bars = axes[0].bar(source_names, mean_weights, yerr=std_weights,
                          color=colors, edgecolor='white', capsize=5)
        axes[0].set_ylabel('平均注意力权重', fontsize=11)
        axes[0].set_title('各数据源平均权重', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, mean_weights):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=11)
        axes[0].set_ylim(0, 1)

        # 2. 权重分布直方图
        for i, (name, color) in enumerate(zip(source_names, colors)):
            axes[1].hist(attention_weights[:, i], bins=50, alpha=0.7,
                        label=name, color=color, edgecolor='white')
        axes[1].set_xlabel('注意力权重', fontsize=11)
        axes[1].set_ylabel('频数', fontsize=11)
        axes[1].set_title('注意力权重分布', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)

        # 3. 散点图（两个源的权重关系）
        if attention_weights.shape[1] >= 2:
            axes[2].scatter(attention_weights[:, 0], attention_weights[:, 1],
                          alpha=0.3, s=10, c=COLORS['info'])
            axes[2].plot([0, 1], [1, 0], 'r--', linewidth=2, label='权重和=1')
            axes[2].set_xlabel(f'{source_names[0]} 权重', fontsize=11)
            axes[2].set_ylabel(f'{source_names[1]} 权重', fontsize=11)
            axes[2].set_title('权重关系散点图', fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_attention_by_class(
        self,
        attention_weights: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        source_names: List[str] = None,
        title: str = "各类别注意力权重",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制各类别的注意力权重分布"""
        if source_names is None:
            source_names = [f'数据源 {i+1}' for i in range(attention_weights.shape[1])]

        n_classes = len(class_names)
        n_sources = len(source_names)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_classes)
        width = 0.8 / n_sources
        colors = PlotStyle.get_color_palette(n_sources)

        for i, (source, color) in enumerate(zip(source_names, colors)):
            mean_weights = []
            std_weights = []
            for cls_idx in range(n_classes):
                cls_mask = labels == cls_idx
                mean_weights.append(np.mean(attention_weights[cls_mask, i]))
                std_weights.append(np.std(attention_weights[cls_mask, i]))

            offset = (i - n_sources/2 + 0.5) * width
            ax.bar(x + offset, mean_weights, width, yerr=std_weights,
                  label=source, color=color, edgecolor='white', capsize=3)

        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('平均注意力权重', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "注意力热力图",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """绘制注意力热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            attention_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=col_labels, yticklabels=row_labels,
            ax=ax, cbar_kws={'label': '注意力权重'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def _save_figure(self, fig: plt.Figure, save_name: Optional[str]):
        if save_name:
            path = os.path.join(self.save_dir, save_name)
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f"图片已保存: {path}")
