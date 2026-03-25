"""
Evaluation module tests.
"""
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import ComprehensiveEvaluator


class TestConfidenceIntervals:
    def test_basic_ci(self):
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 5, n)
        y_pred = y_true.copy()
        noise_idx = np.random.choice(n, int(n * 0.1), replace=False)
        y_pred[noise_idx] = np.random.randint(0, 5, len(noise_idx))

        evaluator = ComprehensiveEvaluator.__new__(ComprehensiveEvaluator)
        evaluator.num_classes = 5
        ci = evaluator.compute_confidence_intervals(y_true, y_pred, n_bootstrap=200, ci=0.95)

        assert "accuracy" in ci
        assert ci["accuracy"]["lower"] <= ci["accuracy"]["mean"] <= ci["accuracy"]["upper"]
        assert ci["accuracy"]["mean"] > 0.8

    def test_ci_with_proba(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 3, n)
        y_pred = y_true.copy()
        y_proba = np.random.dirichlet([5, 1, 1], n)

        evaluator = ComprehensiveEvaluator.__new__(ComprehensiveEvaluator)
        evaluator.num_classes = 3
        ci = evaluator.compute_confidence_intervals(y_true, y_pred, y_proba=y_proba, n_bootstrap=100)
        assert "f1_weighted" in ci


class TestMcNemarTest:
    def test_identical_predictions(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])

        result = ComprehensiveEvaluator.mcnemar_test(y_true, y_pred, y_pred)
        assert result["p_value"] == 1.0
        assert not result["significant"]

    def test_different_predictions(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred_a = y_true.copy()
        y_pred_b = np.random.randint(0, 2, n)

        result = ComprehensiveEvaluator.mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert "statistic" in result
        assert "p_value" in result
        assert isinstance(result["significant"], bool)


class TestOptimalThreshold:
    def test_youden(self):
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1.0, 50)])

        result = ComprehensiveEvaluator.find_optimal_threshold(y_true, y_proba, method="youden")
        assert 0.3 < result["threshold"] < 0.7
        assert result["method"] == "youden"
        assert "tpr" in result
        assert "fpr" in result

    def test_f1_method(self):
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1.0, 50)])

        result = ComprehensiveEvaluator.find_optimal_threshold(y_true, y_proba, method="f1")
        assert result["method"] == "f1"
        assert result["f1"] > 0.5


class TestComprehensiveEvaluator:
    def test_evaluate_with_missing_classes(self):
        class DummyModel(torch.nn.Module):
            def forward(self, *sources):
                labels = sources[-1] if False else None
                batch_size = sources[0].shape[0]
                logits = torch.zeros((batch_size, 3), dtype=torch.float32)
                logits[:, 0] = 2.0
                logits[:, 1] = 1.0
                logits[:, 2] = -5.0
                attention = torch.ones((batch_size, len(sources)), dtype=torch.float32) / len(sources)
                return logits, attention

        n = 24
        s1 = torch.randn(n, 6)
        s2 = torch.randn(n, 4)
        y = torch.randint(0, 2, (n,))
        loader = DataLoader(TensorDataset(s1, s2, y), batch_size=8, shuffle=False)

        evaluator = ComprehensiveEvaluator(
            model=DummyModel(),
            device=torch.device("cpu"),
            class_names=["class0", "class1", "class2"],
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No positive class found in y_true.*")
            results = evaluator.evaluate(loader)

        assert "classification_report" in results
        assert "class2" in results["classification_report"]
        assert len(results["confusion_matrix"]) == 3

    def test_predict_with_three_sources(self):
        class DummyModel(torch.nn.Module):
            def forward(self, *sources):
                batch_size = sources[0].shape[0]
                logits = torch.randn(batch_size, 3)
                attention = torch.ones((batch_size, len(sources)), dtype=torch.float32) / len(sources)
                return logits, attention

        n = 18
        loader = DataLoader(
            TensorDataset(
                torch.randn(n, 5),
                torch.randn(n, 4),
                torch.randn(n, 3),
                torch.randint(0, 3, (n,)),
            ),
            batch_size=6,
            shuffle=False,
        )

        evaluator = ComprehensiveEvaluator(
            model=DummyModel(),
            device=torch.device("cpu"),
            class_names=["a", "b", "c"],
        )
        predictions = evaluator.predict(loader)

        assert predictions["y_true"].shape == (n,)
        assert predictions["y_pred"].shape == (n,)
        assert predictions["y_proba"].shape == (n, 3)
        assert predictions["attention_weights"].shape == (n, 3)
