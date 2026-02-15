"""
评估模块单元测试
"""
import pytest
import numpy as np
from src.evaluation.evaluator import ComprehensiveEvaluator


class TestConfidenceIntervals:
    """测试 Bootstrap 置信区间"""

    def test_basic_ci(self):
        """测试置信区间计算"""
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 5, n)
        # 制造较高准确率的预测
        y_pred = y_true.copy()
        noise_idx = np.random.choice(n, int(n * 0.1), replace=False)
        y_pred[noise_idx] = np.random.randint(0, 5, len(noise_idx))

        evaluator = ComprehensiveEvaluator.__new__(ComprehensiveEvaluator)
        evaluator.num_classes = 5
        ci = evaluator.compute_confidence_intervals(
            y_true, y_pred, n_bootstrap=200, ci=0.95
        )

        assert 'accuracy' in ci
        assert ci['accuracy']['lower'] <= ci['accuracy']['mean'] <= ci['accuracy']['upper']
        assert ci['accuracy']['mean'] > 0.8  # 90% 准确率约

    def test_ci_with_proba(self):
        """测试含概率的置信区间"""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 3, n)
        y_pred = y_true.copy()
        y_proba = np.random.dirichlet([5, 1, 1], n)

        evaluator = ComprehensiveEvaluator.__new__(ComprehensiveEvaluator)
        evaluator.num_classes = 3
        ci = evaluator.compute_confidence_intervals(
            y_true, y_pred, y_proba=y_proba, n_bootstrap=100
        )

        assert 'f1_weighted' in ci


class TestMcNemarTest:
    """测试 McNemar 检验"""

    def test_identical_predictions(self):
        """相同预测应不显著"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])

        result = ComprehensiveEvaluator.mcnemar_test(y_true, y_pred, y_pred)

        assert result['p_value'] == 1.0
        assert not result['significant']

    def test_different_predictions(self):
        """明显不同的预测应显著"""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred_a = y_true.copy()
        y_pred_b = np.random.randint(0, 2, n)

        result = ComprehensiveEvaluator.mcnemar_test(y_true, y_pred_a, y_pred_b)

        assert 'statistic' in result
        assert 'p_value' in result
        assert isinstance(result['significant'], bool)


class TestOptimalThreshold:
    """测试阈值优化"""

    def test_youden(self):
        """测试 Youden's J 方法"""
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([
            np.random.uniform(0, 0.4, 50),
            np.random.uniform(0.6, 1.0, 50)
        ])

        result = ComprehensiveEvaluator.find_optimal_threshold(y_true, y_proba, method='youden')

        assert 0.3 < result['threshold'] < 0.7
        assert result['method'] == 'youden'
        assert 'tpr' in result
        assert 'fpr' in result

    def test_f1_method(self):
        """测试 F1 方法"""
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([
            np.random.uniform(0, 0.4, 50),
            np.random.uniform(0.6, 1.0, 50)
        ])

        result = ComprehensiveEvaluator.find_optimal_threshold(y_true, y_proba, method='f1')

        assert result['method'] == 'f1'
        assert result['f1'] > 0.5
