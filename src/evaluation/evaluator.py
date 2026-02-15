"""
综合模型评估模块

提供完整的模型评估功能，包括：
- 基础分类指标（准确率、精确率、召回率、F1）
- AUC-ROC 和 AUC-PR 曲线
- Bootstrap 置信区间
- McNemar 统计检验（模型对比）
- 阈值优化
- 评估报告生成
"""
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy import stats


class ComprehensiveEvaluator:
    """
    综合模型评估器

    支持多源融合模型的全面评估，包含基础指标、置信区间、
    统计检验和阈值优化等功能。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str],
        output_dir: str = None
    ):
        """
        Args:
            model: 训练好的模型
            device: 计算设备
            class_names: 类别名称列表
            output_dir: 结果保存目录
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def predict(self, data_loader) -> Dict[str, np.ndarray]:
        """
        在数据集上进行预测

        Returns:
            包含 y_true, y_pred, y_proba, attention_weights 的字典
        """
        self.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []
        all_attention = []

        for batch in data_loader:
            if len(batch) == 3:
                s1, s2, labels = batch
                s1 = s1.to(self.device)
                s2 = s2.to(self.device)
                labels = labels.to(self.device)

                output = self.model(s1, s2)
                if isinstance(output, tuple):
                    logits, attention = output
                else:
                    logits = output
                    attention = None
            else:
                raise ValueError(f"不支持的批次格式: 期望3个元素，得到{len(batch)}个")

            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if attention is not None:
                all_attention.extend(attention.cpu().numpy())

        result = {
            'y_true': np.array(all_labels),
            'y_pred': np.array(all_preds),
            'y_proba': np.array(all_probs),
        }

        if all_attention:
            result['attention_weights'] = np.array(all_attention)

        return result

    def evaluate(self, data_loader) -> Dict[str, Any]:
        """
        完整评估流程

        Returns:
            包含所有评估指标的字典
        """
        # 1. 预测
        predictions = self.predict(data_loader)
        y_true = predictions['y_true']
        y_pred = predictions['y_pred']
        y_proba = predictions['y_proba']

        results = {
            'predictions': predictions,
            'class_names': self.class_names,
        }

        # 2. 基础指标
        results['basic_metrics'] = self._compute_basic_metrics(y_true, y_pred, y_proba)

        # 3. 每类指标
        results['per_class_metrics'] = self._compute_per_class_metrics(y_true, y_pred, y_proba)

        # 4. ROC/PR 曲线数据
        results['roc_data'] = self._compute_roc_data(y_true, y_proba)
        results['pr_data'] = self._compute_pr_data(y_true, y_proba)

        # 5. 混淆矩阵
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # 6. 分类报告
        results['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )

        # 7. 置信区间
        results['confidence_intervals'] = self.compute_confidence_intervals(y_true, y_pred, y_proba)

        # 8. 保存结果
        if self.output_dir:
            self._save_results(results)

        return results

    def _compute_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """计算基础分类指标"""
        metrics = {}

        # 准确率
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # 多种平均方式的精确率、召回率、F1
        for avg in ['weighted', 'macro', 'micro']:
            metrics[f'precision_{avg}'] = float(
                precision_score(y_true, y_pred, average=avg, zero_division=0)
            )
            metrics[f'recall_{avg}'] = float(
                recall_score(y_true, y_pred, average=avg, zero_division=0)
            )
            metrics[f'f1_{avg}'] = float(
                f1_score(y_true, y_pred, average=avg, zero_division=0)
            )

        # AUC-ROC
        try:
            if self.num_classes == 2:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics['auc_roc_macro'] = float(
                    roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                )
                metrics['auc_roc_weighted'] = float(
                    roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                )
        except (ValueError, IndexError):
            pass

        # Average Precision
        try:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            if self.num_classes == 2:
                metrics['avg_precision'] = float(
                    average_precision_score(y_true, y_proba[:, 1])
                )
            else:
                metrics['avg_precision_macro'] = float(
                    average_precision_score(y_true_bin, y_proba, average='macro')
                )
        except (ValueError, IndexError):
            pass

        return metrics

    def _compute_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """计算每类指标"""
        per_class = {}
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        for i, name in enumerate(self.class_names):
            mask = y_true == i
            support = int(mask.sum())
            if support == 0:
                continue

            cls_metrics = {
                'support': support,
                'precision': float(precision_score(y_true == i, y_pred == i, zero_division=0)),
                'recall': float(recall_score(y_true == i, y_pred == i, zero_division=0)),
                'f1': float(f1_score(y_true == i, y_pred == i, zero_division=0)),
            }

            # 每类 AUC-ROC
            try:
                if self.num_classes == 2:
                    if i == 1:
                        cls_metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    cls_metrics['auc_roc'] = float(
                        roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                    )
            except (ValueError, IndexError):
                pass

            per_class[name] = cls_metrics

        return per_class

    def _compute_roc_data(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """计算 ROC 曲线数据"""
        roc_data = {}
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        try:
            for i, name in enumerate(self.class_names):
                if self.num_classes == 2:
                    if i == 0:
                        continue
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
                else:
                    fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_proba[:, i])

                roc_auc_val = float(auc(fpr, tpr))
                roc_data[name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc_val
                }
        except (ValueError, IndexError):
            pass

        return roc_data

    def _compute_pr_data(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """计算 PR 曲线数据"""
        pr_data = {}
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        try:
            for i, name in enumerate(self.class_names):
                if self.num_classes == 2:
                    if i == 0:
                        continue
                    prec, rec, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
                else:
                    prec, rec, thresholds = precision_recall_curve(
                        y_true_bin[:, i], y_proba[:, i]
                    )

                pr_auc_val = float(auc(rec, prec))
                pr_data[name] = {
                    'precision': prec.tolist(),
                    'recall': rec.tolist(),
                    'auc': pr_auc_val
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
        ci: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap 置信区间

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            n_bootstrap: Bootstrap 采样次数
            ci: 置信水平

        Returns:
            各指标的置信区间 {metric: {mean, lower, upper}}
        """
        n_samples = len(y_true)
        rng = np.random.RandomState(42)

        acc_scores = []
        f1_scores = []
        auc_scores = []

        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_t = y_true[indices]
            y_p = y_pred[indices]

            # 确保至少两个类别存在
            if len(np.unique(y_t)) < 2:
                continue

            acc_scores.append(accuracy_score(y_t, y_p))
            f1_scores.append(f1_score(y_t, y_p, average='weighted', zero_division=0))

            if y_proba is not None:
                y_prob_boot = y_proba[indices]
                try:
                    if self.num_classes == 2:
                        auc_val = roc_auc_score(y_t, y_prob_boot[:, 1])
                    else:
                        auc_val = roc_auc_score(
                            y_t, y_prob_boot, multi_class='ovr', average='weighted'
                        )
                    auc_scores.append(auc_val)
                except (ValueError, IndexError):
                    pass

        alpha = (1 - ci) / 2
        intervals = {}

        if acc_scores:
            intervals['accuracy'] = {
                'mean': float(np.mean(acc_scores)),
                'lower': float(np.percentile(acc_scores, alpha * 100)),
                'upper': float(np.percentile(acc_scores, (1 - alpha) * 100)),
            }

        if f1_scores:
            intervals['f1_weighted'] = {
                'mean': float(np.mean(f1_scores)),
                'lower': float(np.percentile(f1_scores, alpha * 100)),
                'upper': float(np.percentile(f1_scores, (1 - alpha) * 100)),
            }

        if auc_scores:
            intervals['auc_roc'] = {
                'mean': float(np.mean(auc_scores)),
                'lower': float(np.percentile(auc_scores, alpha * 100)),
                'upper': float(np.percentile(auc_scores, (1 - alpha) * 100)),
            }

        return intervals

    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray
    ) -> Dict[str, float]:
        """
        McNemar 检验：比较两个模型的预测差异是否显著

        Args:
            y_true: 真实标签
            y_pred_a: 模型A的预测
            y_pred_b: 模型B的预测

        Returns:
            包含 statistic 和 p_value 的字典
        """
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # b: A正确B错误; c: A错误B正确
        b = int(np.sum(correct_a & ~correct_b))
        c = int(np.sum(~correct_a & correct_b))

        # McNemar 检验（带连续性校正）
        if b + c == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}

        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

        return {
            'statistic': float(statistic),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'b_count': b,
            'c_count': c
        }

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        method: str = 'youden'
    ) -> Dict[str, Any]:
        """
        寻找最优分类阈值（适用于二分类）

        Args:
            y_true: 真实标签（0/1）
            y_proba: 正类预测概率
            method: 'youden' (Youden's J) 或 'f1'

        Returns:
            包含最优阈值和对应指标的字典
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        if method == 'youden':
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return {
                'threshold': float(thresholds[best_idx]),
                'youden_j': float(j_scores[best_idx]),
                'tpr': float(tpr[best_idx]),
                'fpr': float(fpr[best_idx]),
                'method': 'youden'
            }
        elif method == 'f1':
            precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1s)
            return {
                'threshold': float(pr_thresholds[best_idx]) if best_idx < len(pr_thresholds) else 0.5,
                'f1': float(f1s[best_idx]),
                'precision': float(precisions[best_idx]),
                'recall': float(recalls[best_idx]),
                'method': 'f1'
            }

        raise ValueError(f"不支持的方法: {method}")

    def print_report(self, results: Dict[str, Any]) -> None:
        """打印评估报告"""
        basic = results.get('basic_metrics', {})
        ci = results.get('confidence_intervals', {})

        print("\n" + "=" * 65)
        print("  综合评估报告")
        print("=" * 65)

        # 基础指标
        print(f"\n  准确率:      {basic.get('accuracy', 0):.4f}", end="")
        if 'accuracy' in ci:
            print(f"  [{ci['accuracy']['lower']:.4f}, {ci['accuracy']['upper']:.4f}]")
        else:
            print()

        print(f"  F1 (weighted): {basic.get('f1_weighted', 0):.4f}", end="")
        if 'f1_weighted' in ci:
            print(f"  [{ci['f1_weighted']['lower']:.4f}, {ci['f1_weighted']['upper']:.4f}]")
        else:
            print()

        print(f"  F1 (macro):    {basic.get('f1_macro', 0):.4f}")
        print(f"  F1 (micro):    {basic.get('f1_micro', 0):.4f}")

        if 'auc_roc_macro' in basic:
            print(f"  AUC-ROC (macro): {basic['auc_roc_macro']:.4f}", end="")
            if 'auc_roc' in ci:
                print(f"  [{ci['auc_roc']['lower']:.4f}, {ci['auc_roc']['upper']:.4f}]")
            else:
                print()
        elif 'auc_roc' in basic:
            print(f"  AUC-ROC:       {basic['auc_roc']:.4f}")

        # 每类指标
        per_class = results.get('per_class_metrics', {})
        if per_class:
            print(f"\n  {'类别':<12} {'精确率':>8} {'召回率':>8} {'F1':>8} {'支持数':>8}")
            print("  " + "-" * 50)
            for name, m in per_class.items():
                print(f"  {name:<12} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                      f"{m['f1']:>8.4f} {m['support']:>8d}")

        print("=" * 65)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存评估结果"""
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # 保存完整结果（pickle）
        pkl_path = os.path.join(results_dir, 'evaluation_results.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

        # 保存可读指标（JSON）
        json_metrics = {
            'basic_metrics': results.get('basic_metrics', {}),
            'per_class_metrics': results.get('per_class_metrics', {}),
            'confidence_intervals': results.get('confidence_intervals', {}),
            'confusion_matrix': results.get('confusion_matrix', []),
        }
        json_path = os.path.join(results_dir, 'evaluation_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, ensure_ascii=False, indent=2)

        print(f"评估结果已保存: {results_dir}")
