"""
Model interpretability helpers.
"""
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


class AttentionAnalyzer:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_attention_weights(self, *sources: torch.Tensor) -> Dict[str, np.ndarray]:
        if len(sources) < 2:
            raise ValueError("Attention analysis requires at least two source tensors")

        sources = [source.to(self.device) for source in sources]
        output = self.model(*sources)
        if not isinstance(output, tuple) or len(output) < 2:
            batch_size = sources[0].shape[0]
            num_sources = len(sources)
            return {"fusion_attention": np.ones((batch_size, num_sources)) / num_sources}

        attention = output[1]
        if attention is None:
            batch_size = sources[0].shape[0]
            num_sources = len(sources)
            return {"fusion_attention": np.ones((batch_size, num_sources)) / num_sources}
        if isinstance(attention, dict):
            return {k: v.cpu().numpy() for k, v in attention.items()}
        return {"fusion_attention": attention.cpu().numpy()}

    @torch.no_grad()
    def analyze_by_class(self, dataloader, class_names: List[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        class_attention = defaultdict(list)

        for batch in dataloader:
            *sources, labels = batch
            sources = [source.to(self.device) for source in sources]

            output = self.model(*sources)
            if isinstance(output, tuple) and len(output) >= 2:
                attention = output[1]
            else:
                attention = None

            if attention is None:
                attention = torch.ones(sources[0].shape[0], len(sources), device=self.device) / len(sources)
            if isinstance(attention, dict):
                attention = attention.get("fusion_attention", list(attention.values())[0])

            attention = attention.cpu().numpy()
            for i, label in enumerate(labels.numpy()):
                class_attention[label].append(attention[i])

        results = {}
        for label, attentions in class_attention.items():
            attentions = np.array(attentions)
            class_name = class_names[label] if class_names else str(label)
            results[class_name] = {
                "mean": attentions.mean(axis=0),
                "std": attentions.std(axis=0),
                "median": np.median(attentions, axis=0),
                "count": len(attentions),
            }
        return results

    def plot_attention_distribution(
        self,
        dataloader,
        class_names: List[str] = None,
        save_path: str = None,
        source_names: List[str] = None,
    ) -> plt.Figure:
        class_stats = self.analyze_by_class(dataloader, class_names)
        if not class_stats:
            raise ValueError("No attention statistics available")

        num_sources = len(next(iter(class_stats.values()))["mean"])
        source_names = source_names or [f"Source {i + 1}" for i in range(num_sources)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(source_names))
        width = 0.8 / max(len(class_stats), 1)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_stats), 1)))

        for i, (class_name, stats) in enumerate(class_stats.items()):
            axes[0].bar(x + i * width, stats["mean"], width, label=class_name, alpha=0.8, color=colors[i])

        axes[0].set_xlabel("Data source")
        axes[0].set_ylabel("Attention weight")
        axes[0].set_title("Average attention by class")
        axes[0].set_xticks(x + width * max(len(class_stats) - 1, 0) / 2)
        axes[0].set_xticklabels(source_names, rotation=20)
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        overall_attention = np.mean([stats["mean"] for stats in class_stats.values()], axis=0)
        axes[1].pie(overall_attention, labels=source_names, autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Overall attention distribution")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_attention_heatmap(self, attention_matrix: np.ndarray, source_names: List[str] = None, save_path: str = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        source_names = source_names or [f"Source {i + 1}" for i in range(attention_matrix.shape[0])]
        sns.heatmap(attention_matrix, xticklabels=source_names, yticklabels=source_names, annot=True, fmt=".3f", cmap="Blues", ax=ax)
        ax.set_title("Cross-Attention Weights")
        ax.set_xlabel("Key source")
        ax.set_ylabel("Query source")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


class FeatureImportanceAnalyzer:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def _forward_logits(self, *sources: torch.Tensor) -> torch.Tensor:
        output = self.model(*sources)
        return output[0] if isinstance(output, tuple) else output

    def _compute_loss(self, dataloader, criterion) -> float:
        total_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            *sources, labels = batch
            sources = [source.to(self.device) for source in sources]
            labels = labels.to(self.device)
            outputs = self._forward_logits(*sources)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        return total_loss / total_samples

    def _compute_loss_from_tensors(self, sources: List[torch.Tensor], y: torch.Tensor, criterion) -> float:
        sources = [source.to(self.device) for source in sources]
        y = y.to(self.device)
        outputs = self._forward_logits(*sources)
        return criterion(outputs, y).item()

    @torch.no_grad()
    def permutation_importance(self, dataloader, criterion: nn.Module, n_repeats: int = 5, feature_names: List[str] = None) -> Dict[str, np.ndarray]:
        self.model.eval()
        base_loss = self._compute_loss(dataloader, criterion)

        batch_parts = list(zip(*[batch for batch in dataloader]))
        source_tensors = [torch.cat(parts, dim=0) for parts in batch_parts[:-1]]
        labels = torch.cat(batch_parts[-1], dim=0)

        result = {"base_loss": base_loss}
        for source_idx, source_tensor in enumerate(source_tensors, start=1):
            importance = np.zeros(source_tensor.shape[1])
            for feat_idx in range(source_tensor.shape[1]):
                feat_losses = []
                for repeat_idx in range(n_repeats):
                    permuted_sources = [tensor.clone() for tensor in source_tensors]
                    generator_seed = 42 + source_idx * 1000 + feat_idx * n_repeats + repeat_idx
                    torch.manual_seed(generator_seed)
                    perm_idx = torch.randperm(permuted_sources[source_idx - 1].shape[0])
                    permuted_sources[source_idx - 1][:, feat_idx] = permuted_sources[source_idx - 1][perm_idx, feat_idx]
                    feat_losses.append(self._compute_loss_from_tensors(permuted_sources, labels, criterion))
                importance[feat_idx] = np.mean(feat_losses) - base_loss
            result[f"source{source_idx}_importance"] = importance
        return result

    def gradient_importance(self, *sources: torch.Tensor, target_class: torch.Tensor = None) -> Dict[str, np.ndarray]:
        self.model.eval()
        grad_sources = [source.to(self.device).requires_grad_(True) for source in sources]
        outputs = self._forward_logits(*grad_sources)

        if target_class is None:
            target_class = outputs.argmax(dim=1)
        target_class = target_class.to(self.device)

        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)
        outputs.backward(gradient=one_hot)

        result = {}
        for idx, source in enumerate(grad_sources, start=1):
            result[f"source{idx}_gradient"] = source.grad.abs().mean(dim=0).cpu().numpy()
        return result

    def plot_feature_importance(self, importance: np.ndarray, feature_names: List[str] = None, top_k: int = 20, title: str = "Feature Importance", save_path: str = None) -> plt.Figure:
        feature_names = feature_names or [f"Feature {i}" for i in range(len(importance))]
        indices = np.argsort(importance)[::-1][:top_k]
        top_importance = importance[indices]
        top_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_importance)))
        bars = ax.barh(range(len(top_importance)), top_importance, color=colors)
        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance score")
        ax.set_title(title)

        for bar, val in zip(bars, top_importance):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


class ModelExplainer:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.attention_analyzer = AttentionAnalyzer(model, device)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, device)

    def explain_prediction(
        self,
        *sources: torch.Tensor,
        source_feature_names: Optional[List[List[str]]] = None,
        class_names: List[str] = None,
    ) -> Dict:
        self.model.eval()
        model_sources = [source.to(self.device) for source in sources]

        with torch.no_grad():
            outputs = self.model(*model_sources)
            logits, attention = outputs if isinstance(outputs, tuple) else (outputs, None)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        grad_importance = self.feature_analyzer.gradient_importance(*sources, target_class=torch.tensor([pred_class]))

        explanation = {
            "prediction": {
                "class": pred_class,
                "class_name": class_names[pred_class] if class_names else str(pred_class),
                "confidence": confidence,
                "probabilities": probs[0].cpu().numpy(),
            },
            "attention": attention.cpu().numpy() if isinstance(attention, torch.Tensor) else attention,
            "feature_importance": grad_importance,
        }

        if source_feature_names:
            for source_idx, feature_names in enumerate(source_feature_names, start=1):
                grad_key = f"source{source_idx}_gradient"
                if grad_key not in grad_importance:
                    continue
                top_idx = np.argsort(grad_importance[grad_key])[::-1][:5]
                explanation[f"top_source{source_idx}_features"] = [
                    (feature_names[i], grad_importance[grad_key][i]) for i in top_idx
                ]

        return explanation

    def generate_report(
        self,
        dataloader,
        class_names: List[str] = None,
        source_feature_names: Optional[List[List[str]]] = None,
        output_dir: str = None,
    ) -> Dict:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        attention_stats = self.attention_analyzer.analyze_by_class(dataloader, class_names)
        criterion = nn.CrossEntropyLoss()
        importance = self.feature_analyzer.permutation_importance(dataloader, criterion, n_repeats=3)

        if output_dir:
            fig = self.attention_analyzer.plot_attention_distribution(dataloader, class_names, save_path=os.path.join(output_dir, "attention_distribution.png"))
            plt.close(fig)
            for source_idx in range(1, len([k for k in importance.keys() if k.endswith("_importance")]) + 1):
                key = f"source{source_idx}_importance"
                names = source_feature_names[source_idx - 1] if source_feature_names and source_idx - 1 < len(source_feature_names) else None
                fig = self.feature_analyzer.plot_feature_importance(importance[key], names, title=f"Source {source_idx} Feature Importance", save_path=os.path.join(output_dir, f"source{source_idx}_importance.png"))
                plt.close(fig)

        summary = {
            "total_samples": sum(stats["count"] for stats in attention_stats.values()),
            "num_classes": len(class_names) if class_names else len(attention_stats),
        }
        for source_idx in range(1, len([k for k in importance.keys() if k.endswith("_importance")]) + 1):
            key = f"source{source_idx}_importance"
            summary[f"source{source_idx}_top_features"] = np.argsort(importance[key])[::-1][:10].tolist()

        return {
            "attention_by_class": attention_stats,
            "feature_importance": importance,
            "summary": summary,
        }


def visualize_attention_over_samples(
    model: nn.Module,
    dataloader,
    num_samples: int = 100,
    class_names: List[str] = None,
    save_path: str = None,
) -> plt.Figure:
    device = next(model.parameters()).device
    model.eval()

    all_attention = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break

            *sources, labels = batch
            sources = [source.to(device) for source in sources]
            _, attention = model(*sources)

            if isinstance(attention, dict):
                attention = attention.get("fusion_attention", list(attention.values())[0])

            batch_size = min(labels.size(0), num_samples - count)
            all_attention.extend(attention[:batch_size].cpu().numpy())
            all_labels.extend(labels[:batch_size].numpy())
            count += batch_size

    all_attention = np.array(all_attention)
    all_labels = np.array(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        label_name = class_names[label] if class_names else f"Class {label}"
        if all_attention.shape[1] >= 2:
            axes[0].scatter(all_attention[mask, 0], all_attention[mask, 1], c=[colors[i]], label=label_name, alpha=0.6)

    axes[0].set_xlabel("Source 1 attention")
    axes[0].set_ylabel("Source 2 attention")
    axes[0].set_title("Attention distribution")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if all_attention.shape[1] >= 2:
        source_data = []
        source_labels = []
        source_classes = []
        source_names = [f"Source {i + 1}" for i in range(all_attention.shape[1])]

        for class_label, class_name in zip(unique_labels, [class_names[l] if class_names else f"Class {l}" for l in unique_labels]):
            attn = all_attention[all_labels == class_label]
            for source_idx, source_name in enumerate(source_names):
                source_data.extend(attn[:, source_idx])
                source_labels.extend([source_name] * len(attn))
                source_classes.extend([class_name] * len(attn))

        import pandas as pd

        df = pd.DataFrame({"Attention": source_data, "Source": source_labels, "Class": source_classes})
        sns.boxplot(x="Class", y="Attention", hue="Source", data=df, ax=axes[1])
        axes[1].set_title("Attention by class and source")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
