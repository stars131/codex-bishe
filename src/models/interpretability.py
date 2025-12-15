"""
Model interpretability module for network attack detection.
Provides attention visualization, feature importance analysis, and model explanation tools.

Author: Network Attack Detection Project
"""
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class AttentionAnalyzer:
    """
    注意力权重分析器

    分析和可视化融合网络中的注意力权重。
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化分析器

        Args:
            model: 融合网络模型
            device: 计算设备
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_attention_weights(
        self,
        source1: torch.Tensor,
        source2: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        获取注意力权重

        Args:
            source1: 源1特征 (batch, dim1)
            source2: 源2特征 (batch, dim2)

        Returns:
            包含注意力权重的字典
        """
        source1 = source1.to(self.device)
        source2 = source2.to(self.device)

        _, attention = self.model(source1, source2)

        if isinstance(attention, dict):
            return {k: v.cpu().numpy() for k, v in attention.items()}
        else:
            return {'fusion_attention': attention.cpu().numpy()}

    @torch.no_grad()
    def analyze_by_class(
        self,
        dataloader,
        class_names: List[str] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        按类别分析注意力权重

        Args:
            dataloader: 数据加载器
            class_names: 类别名称

        Returns:
            按类别组织的注意力权重统计
        """
        class_attention = defaultdict(list)

        for batch in dataloader:
            source1, source2, labels = batch
            source1 = source1.to(self.device)
            source2 = source2.to(self.device)

            _, attention = self.model(source1, source2)

            if isinstance(attention, dict):
                attention = attention.get('fusion_attention', list(attention.values())[0])

            attention = attention.cpu().numpy()

            for i, label in enumerate(labels.numpy()):
                class_attention[label].append(attention[i])

        # 计算统计量
        results = {}
        for label, attentions in class_attention.items():
            attentions = np.array(attentions)
            class_name = class_names[label] if class_names else str(label)
            results[class_name] = {
                'mean': attentions.mean(axis=0),
                'std': attentions.std(axis=0),
                'median': np.median(attentions, axis=0),
                'count': len(attentions)
            }

        return results

    def plot_attention_distribution(
        self,
        dataloader,
        class_names: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        绘制注意力权重分布图

        Args:
            dataloader: 数据加载器
            class_names: 类别名称
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        class_stats = self.analyze_by_class(dataloader, class_names)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：各类别的平均注意力权重
        source_names = ['Traffic Features', 'Log Features']
        x = np.arange(len(source_names))
        width = 0.8 / len(class_stats)

        for i, (class_name, stats) in enumerate(class_stats.items()):
            mean_attn = stats['mean']
            if len(mean_attn) >= 2:
                axes[0].bar(x + i * width, mean_attn[:2], width, label=class_name, alpha=0.8)

        axes[0].set_xlabel('Data Source')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Average Attention Weights by Class')
        axes[0].set_xticks(x + width * len(class_stats) / 2)
        axes[0].set_xticklabels(source_names)
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # 右图：注意力权重比例饼图
        overall_attention = []
        for stats in class_stats.values():
            overall_attention.append(stats['mean'][:2] if len(stats['mean']) >= 2 else stats['mean'])

        avg_attention = np.mean(overall_attention, axis=0)
        if len(avg_attention) >= 2:
            axes[1].pie(avg_attention, labels=source_names, autopct='%1.1f%%',
                        colors=['#ff9999', '#66b3ff'], startangle=90)
            axes[1].set_title('Overall Attention Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        source_names: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        绘制注意力热力图

        Args:
            attention_matrix: 注意力矩阵
            source_names: 数据源名称
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if source_names is None:
            source_names = [f'Source {i}' for i in range(attention_matrix.shape[0])]

        sns.heatmap(
            attention_matrix,
            xticklabels=source_names,
            yticklabels=source_names,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            ax=ax
        )

        ax.set_title('Cross-Attention Weights')
        ax.set_xlabel('Key Source')
        ax.set_ylabel('Query Source')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器

    使用多种方法分析特征重要性。
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化分析器

        Args:
            model: 网络模型
            device: 计算设备
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def permutation_importance(
        self,
        dataloader,
        criterion: nn.Module,
        n_repeats: int = 5,
        feature_names: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        置换重要性分析

        通过随机打乱特征来评估其重要性。

        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            n_repeats: 重复次数
            feature_names: 特征名称

        Returns:
            特征重要性字典
        """
        self.model.eval()

        # 计算基准损失
        base_loss = self._compute_loss(dataloader, criterion)

        # 收集所有数据
        all_s1, all_s2, all_y = [], [], []
        for s1, s2, y in dataloader:
            all_s1.append(s1)
            all_s2.append(s2)
            all_y.append(y)

        all_s1 = torch.cat(all_s1, dim=0)
        all_s2 = torch.cat(all_s2, dim=0)
        all_y = torch.cat(all_y, dim=0)

        # 源1特征重要性
        s1_importance = np.zeros(all_s1.shape[1])
        for feat_idx in range(all_s1.shape[1]):
            feat_losses = []
            for _ in range(n_repeats):
                s1_permuted = all_s1.clone()
                perm_idx = torch.randperm(s1_permuted.shape[0])
                s1_permuted[:, feat_idx] = s1_permuted[perm_idx, feat_idx]

                loss = self._compute_loss_from_tensors(s1_permuted, all_s2, all_y, criterion)
                feat_losses.append(loss)

            s1_importance[feat_idx] = np.mean(feat_losses) - base_loss

        # 源2特征重要性
        s2_importance = np.zeros(all_s2.shape[1])
        for feat_idx in range(all_s2.shape[1]):
            feat_losses = []
            for _ in range(n_repeats):
                s2_permuted = all_s2.clone()
                perm_idx = torch.randperm(s2_permuted.shape[0])
                s2_permuted[:, feat_idx] = s2_permuted[perm_idx, feat_idx]

                loss = self._compute_loss_from_tensors(all_s1, s2_permuted, all_y, criterion)
                feat_losses.append(loss)

            s2_importance[feat_idx] = np.mean(feat_losses) - base_loss

        return {
            'source1_importance': s1_importance,
            'source2_importance': s2_importance,
            'base_loss': base_loss
        }

    def _compute_loss(self, dataloader, criterion) -> float:
        """计算数据集上的平均损失"""
        total_loss = 0.0
        total_samples = 0

        for s1, s2, y in dataloader:
            s1, s2, y = s1.to(self.device), s2.to(self.device), y.to(self.device)
            outputs, _ = self.model(s1, s2)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        return total_loss / total_samples

    def _compute_loss_from_tensors(self, s1, s2, y, criterion) -> float:
        """从张量计算损失"""
        s1, s2, y = s1.to(self.device), s2.to(self.device), y.to(self.device)
        outputs, _ = self.model(s1, s2)
        loss = criterion(outputs, y)
        return loss.item()

    def gradient_importance(
        self,
        source1: torch.Tensor,
        source2: torch.Tensor,
        target_class: int = None
    ) -> Dict[str, np.ndarray]:
        """
        基于梯度的特征重要性

        Args:
            source1: 源1特征
            source2: 源2特征
            target_class: 目标类别（None表示预测类别）

        Returns:
            特征重要性字典
        """
        self.model.eval()

        source1 = source1.to(self.device).requires_grad_(True)
        source2 = source2.to(self.device).requires_grad_(True)

        outputs, _ = self.model(source1, source2)

        if target_class is None:
            target_class = outputs.argmax(dim=1)

        # 计算梯度
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)

        outputs.backward(gradient=one_hot)

        s1_grad = source1.grad.abs().mean(dim=0).cpu().numpy()
        s2_grad = source2.grad.abs().mean(dim=0).cpu().numpy()

        return {
            'source1_gradient': s1_grad,
            'source2_gradient': s2_grad
        }

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str] = None,
        top_k: int = 20,
        title: str = 'Feature Importance',
        save_path: str = None
    ) -> plt.Figure:
        """
        绘制特征重要性条形图

        Args:
            importance: 重要性数组
            feature_names: 特征名称
            top_k: 显示前k个特征
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]

        # 排序
        indices = np.argsort(importance)[::-1][:top_k]
        top_importance = importance[indices]
        top_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))

        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_importance)))
        bars = ax.barh(range(len(top_importance)), top_importance, color=colors)

        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)

        # 添加数值标签
        for bar, val in zip(bars, top_importance):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping

    用于可视化模型关注的特征区域。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        初始化GradCAM

        Args:
            model: 网络模型
            target_layer: 目标层
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        source1: torch.Tensor,
        source2: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        生成CAM

        Args:
            source1: 源1特征
            source2: 源2特征
            target_class: 目标类别

        Returns:
            CAM热力图
        """
        self.model.eval()

        outputs, _ = self.model(source1, source2)

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        weights = self.gradients.mean(dim=(0, 2, 3) if self.gradients.dim() == 4 else (0,))

        # 加权激活
        cam = (weights.unsqueeze(-1) * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


class ModelExplainer:
    """
    综合模型解释器

    整合多种解释方法。
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化解释器

        Args:
            model: 网络模型
            device: 计算设备
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.attention_analyzer = AttentionAnalyzer(model, device)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, device)

    def explain_prediction(
        self,
        source1: torch.Tensor,
        source2: torch.Tensor,
        source1_names: List[str] = None,
        source2_names: List[str] = None,
        class_names: List[str] = None
    ) -> Dict:
        """
        解释单个预测

        Args:
            source1: 源1特征
            source2: 源2特征
            source1_names: 源1特征名称
            source2_names: 源2特征名称
            class_names: 类别名称

        Returns:
            解释结果字典
        """
        self.model.eval()

        source1 = source1.to(self.device)
        source2 = source2.to(self.device)

        # 预测
        with torch.no_grad():
            outputs, attention = self.model(source1, source2)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        # 获取梯度重要性
        grad_importance = self.feature_analyzer.gradient_importance(
            source1, source2, torch.tensor([pred_class])
        )

        explanation = {
            'prediction': {
                'class': pred_class,
                'class_name': class_names[pred_class] if class_names else str(pred_class),
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy()
            },
            'attention': attention.cpu().numpy() if not isinstance(attention, dict)
                        else {k: v.cpu().numpy() for k, v in attention.items()},
            'feature_importance': {
                'source1': grad_importance['source1_gradient'],
                'source2': grad_importance['source2_gradient']
            }
        }

        # 添加特征名称
        if source1_names:
            top_s1_idx = np.argsort(grad_importance['source1_gradient'])[::-1][:5]
            explanation['top_source1_features'] = [
                (source1_names[i], grad_importance['source1_gradient'][i])
                for i in top_s1_idx
            ]

        if source2_names:
            top_s2_idx = np.argsort(grad_importance['source2_gradient'])[::-1][:5]
            explanation['top_source2_features'] = [
                (source2_names[i], grad_importance['source2_gradient'][i])
                for i in top_s2_idx
            ]

        return explanation

    def generate_report(
        self,
        dataloader,
        class_names: List[str] = None,
        source1_names: List[str] = None,
        source2_names: List[str] = None,
        output_dir: str = None
    ) -> Dict:
        """
        生成完整的解释报告

        Args:
            dataloader: 数据加载器
            class_names: 类别名称
            source1_names: 源1特征名称
            source2_names: 源2特征名称
            output_dir: 输出目录

        Returns:
            报告字典
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        report = {}

        # 1. 注意力分析
        print("分析注意力权重...")
        attention_stats = self.attention_analyzer.analyze_by_class(dataloader, class_names)
        report['attention_by_class'] = attention_stats

        if output_dir:
            fig = self.attention_analyzer.plot_attention_distribution(
                dataloader, class_names,
                save_path=os.path.join(output_dir, 'attention_distribution.png')
            )
            plt.close(fig)

        # 2. 特征重要性分析
        print("分析特征重要性...")
        criterion = nn.CrossEntropyLoss()
        importance = self.feature_analyzer.permutation_importance(
            dataloader, criterion, n_repeats=3
        )
        report['feature_importance'] = importance

        if output_dir:
            fig = self.feature_analyzer.plot_feature_importance(
                importance['source1_importance'],
                source1_names,
                title='Source 1 Feature Importance',
                save_path=os.path.join(output_dir, 'source1_importance.png')
            )
            plt.close(fig)

            fig = self.feature_analyzer.plot_feature_importance(
                importance['source2_importance'],
                source2_names,
                title='Source 2 Feature Importance',
                save_path=os.path.join(output_dir, 'source2_importance.png')
            )
            plt.close(fig)

        # 3. 汇总统计
        report['summary'] = {
            'total_samples': sum(s['count'] for s in attention_stats.values()),
            'num_classes': len(class_names) if class_names else len(attention_stats),
            'source1_top_features': np.argsort(importance['source1_importance'])[::-1][:10].tolist(),
            'source2_top_features': np.argsort(importance['source2_importance'])[::-1][:10].tolist()
        }

        return report


def visualize_attention_over_samples(
    model: nn.Module,
    dataloader,
    num_samples: int = 100,
    class_names: List[str] = None,
    save_path: str = None
) -> plt.Figure:
    """
    可视化多个样本的注意力权重

    Args:
        model: 网络模型
        dataloader: 数据加载器
        num_samples: 样本数量
        class_names: 类别名称
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    device = next(model.parameters()).device
    model.eval()

    all_attention = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for s1, s2, labels in dataloader:
            if count >= num_samples:
                break

            s1, s2 = s1.to(device), s2.to(device)
            _, attention = model(s1, s2)

            if isinstance(attention, dict):
                attention = list(attention.values())[0]

            batch_size = min(s1.size(0), num_samples - count)
            all_attention.extend(attention[:batch_size].cpu().numpy())
            all_labels.extend(labels[:batch_size].numpy())
            count += batch_size

    all_attention = np.array(all_attention)
    all_labels = np.array(all_labels)

    # 创建图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：按类别着色的散点图
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        label_name = class_names[label] if class_names else f'Class {label}'

        if all_attention.shape[1] >= 2:
            axes[0].scatter(
                all_attention[mask, 0],
                all_attention[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.6
            )

    axes[0].set_xlabel('Traffic Feature Attention')
    axes[0].set_ylabel('Log Feature Attention')
    axes[0].set_title('Attention Weights Distribution')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # 右图：各类别的箱线图
    attention_by_class = {
        (class_names[l] if class_names else f'Class {l}'): all_attention[all_labels == l]
        for l in unique_labels
    }

    if all_attention.shape[1] >= 2:
        source_data = []
        source_labels = []
        source_classes = []

        for class_name, attn in attention_by_class.items():
            source_data.extend(attn[:, 0])
            source_labels.extend(['Traffic'] * len(attn))
            source_classes.extend([class_name] * len(attn))

            source_data.extend(attn[:, 1])
            source_labels.extend(['Log'] * len(attn))
            source_classes.extend([class_name] * len(attn))

        import pandas as pd
        df = pd.DataFrame({
            'Attention': source_data,
            'Source': source_labels,
            'Class': source_classes
        })

        sns.boxplot(x='Class', y='Attention', hue='Source', data=df, ax=axes[1])
        axes[1].set_title('Attention Distribution by Class and Source')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
