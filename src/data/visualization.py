"""
数据分析和可视化工具

用于探索性数据分析、特征分析和结果可视化。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataAnalyzer:
    """
    数据分析器

    提供数据探索、统计分析和可视化功能。
    """

    def __init__(self, save_dir: str = "outputs/figures"):
        """
        初始化分析器

        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def class_distribution(
        self,
        labels: np.ndarray,
        class_names: List[str],
        title: str = "Class Distribution",
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制类别分布图

        Args:
            labels: 标签数组
            class_names: 类别名称列表
            title: 图表标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 计算分布
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100

        # 柱状图
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        bars = axes[0].bar(range(len(unique)), counts, color=colors)
        axes[0].set_xticks(range(len(unique)))
        axes[0].set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
        axes[0].set_ylabel('Sample Count')
        axes[0].set_title(f'{title} - Bar Chart')

        # 添加数值标签
        for bar, count, pct in zip(bars, counts, percentages):
            axes[0].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8
            )

        # 饼图
        axes[1].pie(
            counts,
            labels=[f'{class_names[i]}\n({c:,})' for i, c in zip(unique, counts)],
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(unique)
        )
        axes[1].set_title(f'{title} - Pie Chart')

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def feature_statistics(
        self,
        features: np.ndarray,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        计算特征统计信息

        Args:
            features: 特征数组
            feature_names: 特征名称列表
            top_n: 显示前N个特征

        Returns:
            统计信息DataFrame
        """
        stats = pd.DataFrame({
            'Feature': feature_names,
            'Mean': np.mean(features, axis=0),
            'Std': np.std(features, axis=0),
            'Min': np.min(features, axis=0),
            'Max': np.max(features, axis=0),
            'Median': np.median(features, axis=0),
            'Variance': np.var(features, axis=0)
        })

        stats = stats.sort_values('Variance', ascending=False)
        print(f"\n特征统计 (Top {top_n}):")
        print(stats.head(top_n).to_string(index=False))

        return stats

    def correlation_matrix(
        self,
        features: np.ndarray,
        feature_names: List[str],
        top_n: int = 30,
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制特征相关性矩阵

        Args:
            features: 特征数组
            feature_names: 特征名称列表
            top_n: 显示前N个特征
            save_name: 保存文件名
        """
        # 选择方差最大的特征
        variances = np.var(features, axis=0)
        top_indices = np.argsort(variances)[-top_n:]

        selected_features = features[:, top_indices]
        selected_names = [feature_names[i] for i in top_indices]

        # 计算相关性
        corr_matrix = np.corrcoef(selected_features.T)

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            xticklabels=selected_names,
            yticklabels=selected_names,
            cmap='RdBu_r',
            center=0,
            annot=False,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title(f'Feature Correlation Matrix (Top {top_n} by Variance)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def feature_distribution_by_class(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        feature_indices: List[int] = None,
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制各类别的特征分布

        Args:
            features: 特征数组
            labels: 标签数组
            feature_names: 特征名称列表
            class_names: 类别名称列表
            feature_indices: 要绘制的特征索引
            save_name: 保存文件名
        """
        if feature_indices is None:
            # 默认选择方差最大的4个特征
            variances = np.var(features, axis=0)
            feature_indices = np.argsort(variances)[-4:]

        n_features = len(feature_indices)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            feat_name = feature_names[feat_idx]

            for cls_idx in np.unique(labels):
                cls_data = features[labels == cls_idx, feat_idx]
                ax.hist(
                    cls_data,
                    bins=50,
                    alpha=0.5,
                    label=class_names[cls_idx],
                    density=True
                )

            ax.set_xlabel(feat_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feat_name}')
            ax.legend(fontsize=8)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def dimensionality_reduction_plot(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        method: str = 'tsne',
        n_samples: int = 5000,
        save_name: Optional[str] = None
    ) -> None:
        """
        降维可视化

        Args:
            features: 特征数组
            labels: 标签数组
            class_names: 类别名称列表
            method: 降维方法 ('tsne', 'pca')
            n_samples: 采样数量
            save_name: 保存文件名
        """
        # 采样（避免计算过慢）
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels

        print(f"执行 {method.upper()} 降维 ({len(features_sample)} 样本)...")

        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)

        reduced = reducer.fit_transform(features_sample)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels_sample))))
        for cls_idx in np.unique(labels_sample):
            mask = labels_sample == cls_idx
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[cls_idx]],
                label=class_names[cls_idx],
                alpha=0.6,
                s=20
            )

        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'{method.upper()} Visualization of Network Traffic')
        ax.legend(loc='best', fontsize=8)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def boxplot_by_class(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        feature_indices: List[int] = None,
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制各类别的特征箱线图

        Args:
            features: 特征数组
            labels: 标签数组
            feature_names: 特征名称列表
            class_names: 类别名称列表
            feature_indices: 要绘制的特征索引
            save_name: 保存文件名
        """
        if feature_indices is None:
            variances = np.var(features, axis=0)
            feature_indices = np.argsort(variances)[-6:]

        n_features = len(feature_indices)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            feat_name = feature_names[feat_idx]

            data = [features[labels == cls_idx, feat_idx] for cls_idx in np.unique(labels)]
            bp = ax.boxplot(data, labels=[class_names[idx] for idx in np.unique(labels)])
            ax.set_title(feat_name)
            ax.tick_params(axis='x', rotation=45)

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()


class ResultVisualizer:
    """
    结果可视化器

    用于训练过程和模型评估结果的可视化。
    """

    def __init__(self, save_dir: str = "outputs/figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制训练曲线

        Args:
            history: 训练历史字典 {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss曲线
        axes[0].plot(history.get('train_loss', []), label='Train Loss', linewidth=2)
        axes[0].plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy曲线
        axes[1].plot(history.get('train_acc', []), label='Train Accuracy', linewidth=2)
        axes[1].plot(history.get('val_acc', []), label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制混淆矩阵

        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            normalize: 是否归一化
            title: 图表标题
            save_name: 保存文件名
        """
        if normalize:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            cm_plot = cm
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def plot_roc_curves(
        self,
        fpr_dict: Dict[str, np.ndarray],
        tpr_dict: Dict[str, np.ndarray],
        auc_dict: Dict[str, float],
        title: str = "ROC Curves",
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制ROC曲线

        Args:
            fpr_dict: 各类别的FPR字典
            tpr_dict: 各类别的TPR字典
            auc_dict: 各类别的AUC字典
            title: 图表标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(fpr_dict)))

        for i, (class_name, fpr) in enumerate(fpr_dict.items()):
            tpr = tpr_dict[class_name]
            auc = auc_dict[class_name]
            ax.plot(
                fpr, tpr,
                color=colors[i],
                linewidth=2,
                label=f'{class_name} (AUC = {auc:.4f})'
            )

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def plot_precision_recall_curves(
        self,
        precision_dict: Dict[str, np.ndarray],
        recall_dict: Dict[str, np.ndarray],
        ap_dict: Dict[str, float],
        title: str = "Precision-Recall Curves",
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制PR曲线

        Args:
            precision_dict: 各类别的Precision字典
            recall_dict: 各类别的Recall字典
            ap_dict: 各类别的AP字典
            title: 图表标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(precision_dict)))

        for i, (class_name, precision) in enumerate(precision_dict.items()):
            recall = recall_dict[class_name]
            ap = ap_dict[class_name]
            ax.plot(
                recall, precision,
                color=colors[i],
                linewidth=2,
                label=f'{class_name} (AP = {ap:.4f})'
            )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_names: List[str] = None,
        title: str = "Model Performance Comparison",
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制模型性能对比图

        Args:
            metrics_dict: {模型名: {指标名: 值}} 的字典
            metric_names: 要比较的指标列表
            title: 图表标题
            save_name: 保存文件名
        """
        if metric_names is None:
            metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        model_names = list(metrics_dict.keys())
        n_models = len(model_names)
        n_metrics = len(metric_names)

        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name].get(m, 0) for m in metric_names]
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height(),
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=0
                )

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        source_names: List[str] = None,
        class_names: List[str] = None,
        title: str = "Attention Weights Analysis",
        save_name: Optional[str] = None
    ) -> None:
        """
        绘制注意力权重分析图

        Args:
            attention_weights: 注意力权重数组 (n_samples, n_sources) 或 (n_classes, n_sources)
            source_names: 数据源名称列表
            class_names: 类别名称列表（如果按类别分析）
            title: 图表标题
            save_name: 保存文件名
        """
        if source_names is None:
            source_names = [f'Source {i+1}' for i in range(attention_weights.shape[1])]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 平均注意力权重
        mean_weights = np.mean(attention_weights, axis=0)
        axes[0].bar(source_names, mean_weights, color=['#3498db', '#e74c3c'])
        axes[0].set_ylabel('Average Attention Weight')
        axes[0].set_title('Average Attention Weights per Source')
        for i, v in enumerate(mean_weights):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')

        # 注意力权重分布
        for i, name in enumerate(source_names):
            axes[1].hist(
                attention_weights[:, i],
                bins=50,
                alpha=0.6,
                label=name
            )
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Attention Weight Distribution')
        axes[1].legend()

        plt.suptitle(title)
        plt.tight_layout()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_name}")

        plt.show()


def generate_data_report(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    save_dir: str = "outputs/figures"
) -> None:
    """
    生成完整的数据分析报告

    Args:
        features: 特征数组
        labels: 标签数组
        feature_names: 特征名称列表
        class_names: 类别名称列表
        save_dir: 保存目录
    """
    print("=" * 60)
    print("生成数据分析报告")
    print("=" * 60)

    analyzer = DataAnalyzer(save_dir)

    # 1. 基本统计
    print(f"\n数据集概览:")
    print(f"  样本数: {len(labels):,}")
    print(f"  特征数: {len(feature_names)}")
    print(f"  类别数: {len(class_names)}")

    # 2. 类别分布
    print("\n绘制类别分布...")
    analyzer.class_distribution(
        labels, class_names,
        title="Attack Type Distribution",
        save_name="class_distribution.png"
    )

    # 3. 特征统计
    print("\n计算特征统计...")
    stats = analyzer.feature_statistics(features, feature_names, top_n=15)

    # 4. 相关性矩阵
    print("\n绘制相关性矩阵...")
    analyzer.correlation_matrix(
        features, feature_names,
        top_n=25,
        save_name="correlation_matrix.png"
    )

    # 5. 特征分布
    print("\n绘制特征分布...")
    analyzer.feature_distribution_by_class(
        features, labels, feature_names, class_names,
        save_name="feature_distribution.png"
    )

    # 6. 降维可视化
    print("\n绘制降维可视化...")
    analyzer.dimensionality_reduction_plot(
        features, labels, class_names,
        method='tsne',
        n_samples=5000,
        save_name="tsne_visualization.png"
    )

    print("\n报告生成完成!")
    print(f"图片保存在: {save_dir}")
