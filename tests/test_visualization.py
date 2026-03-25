"""
Visualization and report tests.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.interpretability import AttentionAnalyzer
from src.visualization.plots import AttentionVisualizer
from src.visualization.report import generate_full_report


class DummyFusionModel(torch.nn.Module):
    def forward(self, *sources):
        batch_size = sources[0].shape[0]
        logits = torch.randn(batch_size, 3)
        attention = torch.ones((batch_size, len(sources)), dtype=torch.float32) / len(sources)
        return logits, attention


def test_attention_visualizer_supports_three_sources(tmp_path):
    attention_weights = np.random.dirichlet([1, 1, 1], size=128)
    labels = np.random.randint(0, 3, size=128)
    class_names = ["a", "b", "c"]
    source_names = ["s1", "s2", "s3"]

    viz = AttentionVisualizer(str(tmp_path))
    fig1 = viz.plot_attention_weights(attention_weights, source_names, save_name="attn.png")
    fig2 = viz.plot_attention_by_class(attention_weights, labels, class_names, source_names, save_name="attn_by_class.png")

    assert fig1 is not None
    assert fig2 is not None
    assert (tmp_path / "attn.png").exists()
    assert (tmp_path / "attn_by_class.png").exists()


def test_attention_analyzer_supports_three_sources():
    model = DummyFusionModel()
    loader = DataLoader(
        TensorDataset(
            torch.randn(12, 4),
            torch.randn(12, 5),
            torch.randn(12, 6),
            torch.randint(0, 3, (12,)),
        ),
        batch_size=4,
        shuffle=False,
    )

    analyzer = AttentionAnalyzer(model, device=torch.device("cpu"))
    stats = analyzer.analyze_by_class(loader, class_names=["a", "b", "c"])

    assert stats
    for item in stats.values():
        assert len(item["mean"]) == 3


def test_generate_full_report_preserves_multi_source_names(tmp_path):
    results_path = tmp_path / "results.pkl"
    history_path = tmp_path / "history.pkl"

    results = {
        "y_true": np.array([0, 1, 2, 0]),
        "y_pred": np.array([0, 1, 1, 0]),
        "y_proba": np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
        ]),
        "class_names": ["a", "b", "c"],
        "metrics": {"accuracy": 0.75, "precision_weighted": 0.8, "recall_weighted": 0.75, "f1_weighted": 0.74},
        "attention_weights": np.random.dirichlet([1, 1, 1], size=4),
        "source_names": ["alpha", "beta", "gamma"],
    }
    history = {"history": {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6], "val_acc": [0.4, 0.7]}}

    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    report_path = generate_full_report(
        experiment_name="demo",
        results_path=str(results_path),
        history_path=str(history_path),
        output_dir=str(tmp_path / "reports"),
    )

    report_data_path = tmp_path / "reports" / "demo" / "report_data.pkl"
    with open(report_data_path, "rb") as f:
        report_data = pickle.load(f)

    assert report_path.endswith("report.html")
    assert report_data["sections"]["attention"]["statistics"]["source_names"] == ["alpha", "beta", "gamma"]
