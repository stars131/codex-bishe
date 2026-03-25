"""
Comprehensive evaluation utilities for classification models.
"""
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


class ComprehensiveEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str],
        output_dir: str = None,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def predict(self, data_loader) -> Dict[str, np.ndarray]:
        self.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []
        all_attention = []
        all_agentic_actions = []

        for batch in data_loader:
            if len(batch) < 2:
                raise ValueError(f"Unsupported batch format: expected features plus labels, got {len(batch)} items")

            *features, labels = batch
            features = [feature.to(self.device) for feature in features]
            labels = labels.to(self.device)

            output = self.model(*features) if len(features) > 1 else self.model(features[0])
            if isinstance(output, tuple):
                logits, attention = output
            else:
                logits = output
                attention = None

            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if attention is not None:
                all_attention.extend(attention.cpu().numpy())
            actions = getattr(self.model, "last_agentic_actions", None)
            if actions:
                all_agentic_actions.extend(actions)

        result = {
            "y_true": np.array(all_labels),
            "y_pred": np.array(all_preds),
            "y_proba": np.array(all_probs),
        }
        if all_attention:
            result["attention_weights"] = np.array(all_attention)
        if all_agentic_actions:
            result["agentic_actions"] = np.array(all_agentic_actions)
        return result

    def evaluate(self, data_loader) -> Dict[str, Any]:
        predictions = self.predict(data_loader)
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        y_proba = predictions["y_proba"]

        results = {
            "predictions": predictions,
            "class_names": self.class_names,
            "basic_metrics": self._compute_basic_metrics(y_true, y_pred, y_proba),
            "per_class_metrics": self._compute_per_class_metrics(y_true, y_pred, y_proba),
            "roc_data": self._compute_roc_data(y_true, y_proba),
            "pr_data": self._compute_pr_data(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes))).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=list(range(self.num_classes)),
                target_names=self.class_names,
                zero_division=0,
                output_dict=True,
            ),
            "confidence_intervals": self.compute_confidence_intervals(y_true, y_pred, y_proba),
        }

        if self.output_dir:
            self._save_results(results)
        return results

    def _compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}

        for avg in ["weighted", "macro", "micro"]:
            metrics[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
            metrics[f"recall_{avg}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
            metrics[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

        try:
            if self.num_classes == 2:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc_roc_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
                metrics["auc_roc_weighted"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except (ValueError, IndexError):
            pass

        try:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            if self.num_classes == 2:
                metrics["avg_precision"] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics["avg_precision_macro"] = float(average_precision_score(y_true_bin, y_proba, average="macro"))
        except (ValueError, IndexError):
            pass

        return metrics

    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
        per_class = {}
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        for i, name in enumerate(self.class_names):
            mask = y_true == i
            support = int(mask.sum())
            if support == 0:
                continue

            cls_metrics = {
                "support": support,
                "precision": float(precision_score(y_true == i, y_pred == i, zero_division=0)),
                "recall": float(recall_score(y_true == i, y_pred == i, zero_division=0)),
                "f1": float(f1_score(y_true == i, y_pred == i, zero_division=0)),
            }

            try:
                if self.num_classes == 2:
                    if i == 1:
                        cls_metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    cls_metrics["auc_roc"] = float(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))
            except (ValueError, IndexError):
                pass

            per_class[name] = cls_metrics

        return per_class

    def _compute_roc_data(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        roc_data = {}
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        try:
            if len(np.unique(y_true)) < 2:
                return roc_data
            for i, name in enumerate(self.class_names):
                if self.num_classes == 2:
                    if i == 0:
                        continue
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                else:
                    if len(np.unique(y_true_bin[:, i])) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_data[name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc(fpr, tpr)),
                }
        except (ValueError, IndexError):
            pass

        return roc_data

    def _compute_pr_data(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        pr_data = {}
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        try:
            if len(np.unique(y_true)) < 2:
                return pr_data
            for i, name in enumerate(self.class_names):
                if self.num_classes == 2:
                    if i == 0:
                        continue
                    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                else:
                    if y_true_bin[:, i].sum() == 0:
                        continue
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                pr_data[name] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "auc": float(auc(recall, precision)),
                }
        except (ValueError, IndexError):
            pass

        return pr_data

    def compute_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        n_samples = len(y_true)
        rng = np.random.RandomState(42)

        acc_scores = []
        f1_scores = []
        auc_scores = []

        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_t = y_true[indices]
            y_p = y_pred[indices]

            if len(np.unique(y_t)) < 2:
                continue

            acc_scores.append(accuracy_score(y_t, y_p))
            f1_scores.append(f1_score(y_t, y_p, average="weighted", zero_division=0))

            if y_proba is not None:
                y_prob_boot = y_proba[indices]
                try:
                    if self.num_classes == 2:
                        auc_val = roc_auc_score(y_t, y_prob_boot[:, 1])
                    else:
                        auc_val = roc_auc_score(y_t, y_prob_boot, multi_class="ovr", average="weighted")
                    auc_scores.append(auc_val)
                except (ValueError, IndexError):
                    pass

        alpha = (1 - ci) / 2
        intervals = {}

        if acc_scores:
            intervals["accuracy"] = {
                "mean": float(np.mean(acc_scores)),
                "lower": float(np.percentile(acc_scores, alpha * 100)),
                "upper": float(np.percentile(acc_scores, (1 - alpha) * 100)),
            }
        if f1_scores:
            intervals["f1_weighted"] = {
                "mean": float(np.mean(f1_scores)),
                "lower": float(np.percentile(f1_scores, alpha * 100)),
                "upper": float(np.percentile(f1_scores, (1 - alpha) * 100)),
            }
        if auc_scores:
            intervals["auc_roc"] = {
                "mean": float(np.mean(auc_scores)),
                "lower": float(np.percentile(auc_scores, alpha * 100)),
                "upper": float(np.percentile(auc_scores, (1 - alpha) * 100)),
            }

        return intervals

    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
        correct_a = y_pred_a == y_true
        correct_b = y_pred_b == y_true
        b = int(np.sum(correct_a & ~correct_b))
        c = int(np.sum(~correct_a & correct_b))

        if b + c == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))
        return {
            "statistic": float(statistic),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "b_count": b,
            "c_count": c,
        }

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, method: str = "youden") -> Dict[str, Any]:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        if method == "youden":
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return {
                "threshold": float(thresholds[best_idx]),
                "youden_j": float(j_scores[best_idx]),
                "tpr": float(tpr[best_idx]),
                "fpr": float(fpr[best_idx]),
                "method": "youden",
            }
        if method == "f1":
            precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1s)
            return {
                "threshold": float(pr_thresholds[best_idx]) if best_idx < len(pr_thresholds) else 0.5,
                "f1": float(f1s[best_idx]),
                "precision": float(precisions[best_idx]),
                "recall": float(recalls[best_idx]),
                "method": "f1",
            }
        raise ValueError(f"Unsupported method: {method}")

    def print_report(self, results: Dict[str, Any]) -> None:
        basic = results.get("basic_metrics", {})
        ci = results.get("confidence_intervals", {})

        print("\n" + "=" * 65)
        print("  Evaluation Report")
        print("=" * 65)

        print(f"\n  Accuracy:      {basic.get('accuracy', 0):.4f}", end="")
        if "accuracy" in ci:
            print(f"  [{ci['accuracy']['lower']:.4f}, {ci['accuracy']['upper']:.4f}]")
        else:
            print()

        print(f"  F1 (weighted): {basic.get('f1_weighted', 0):.4f}", end="")
        if "f1_weighted" in ci:
            print(f"  [{ci['f1_weighted']['lower']:.4f}, {ci['f1_weighted']['upper']:.4f}]")
        else:
            print()

        print(f"  F1 (macro):    {basic.get('f1_macro', 0):.4f}")
        print(f"  F1 (micro):    {basic.get('f1_micro', 0):.4f}")

        if "auc_roc_macro" in basic:
            print(f"  AUC-ROC (macro): {basic['auc_roc_macro']:.4f}", end="")
            if "auc_roc" in ci:
                print(f"  [{ci['auc_roc']['lower']:.4f}, {ci['auc_roc']['upper']:.4f}]")
            else:
                print()
        elif "auc_roc" in basic:
            print(f"  AUC-ROC:       {basic['auc_roc']:.4f}")

        per_class = results.get("per_class_metrics", {})
        if per_class:
            print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
            print("  " + "-" * 55)
            for name, metrics in per_class.items():
                print(
                    f"  {name:<12} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} "
                    f"{metrics['f1']:>8.4f} {metrics['support']:>8d}"
                )

        print("=" * 65)

    def _save_results(self, results: Dict[str, Any]) -> None:
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "evaluation_results.pkl"), "wb") as f:
            pickle.dump(results, f)

        json_metrics = {
            "basic_metrics": results.get("basic_metrics", {}),
            "per_class_metrics": results.get("per_class_metrics", {}),
            "confidence_intervals": results.get("confidence_intervals", {}),
            "confusion_matrix": results.get("confusion_matrix", []),
        }
        with open(os.path.join(results_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(json_metrics, f, ensure_ascii=False, indent=2)
