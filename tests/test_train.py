"""
Training module tests.
"""
import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.losses import ClassBalancedLoss
from src.train import Trainer


class DummyClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def make_logger() -> logging.Logger:
    logger = logging.getLogger("test-trainer")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.NullHandler())
    return logger


def make_config(loss_type: str = "cross_entropy", monitor: str = "val_loss", epochs: int = 2) -> dict:
    return {
        "model": {
            "num_classes": 3,
        },
        "training": {
            "epochs": epochs,
            "batch_size": 4,
            "mixed_precision": False,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "type": "none",
                "warmup_epochs": 0,
                "min_lr": 1e-6,
            },
            "loss": {
                "type": loss_type,
                "class_weights": None,
                "label_smoothing": 0.0,
            },
            "early_stopping": {
                "enabled": False,
                "patience": 10,
                "min_delta": 0.0,
                "monitor": monitor,
            },
            "checkpoint": {
                "save_every": 100,
            },
        },
    }


def test_trainer_initializes_class_balanced_loss(tmp_path):
    features = torch.randn(9, 5)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=3, shuffle=False)

    trainer = Trainer(
        model=DummyClassifier(input_dim=5, num_classes=3),
        train_loader=loader,
        val_loader=loader,
        config=make_config(loss_type="class_balanced"),
        device=torch.device("cpu"),
        logger=make_logger(),
        output_dir=str(tmp_path),
    )

    assert isinstance(trainer.criterion, ClassBalancedLoss)
    assert trainer.criterion.samples_per_class == [4, 3, 2]


def test_trainer_uses_configured_monitor_for_best_checkpoint(tmp_path):
    features = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)

    trainer = Trainer(
        model=DummyClassifier(input_dim=4, num_classes=3),
        train_loader=loader,
        val_loader=loader,
        config=make_config(monitor="val_f1_macro", epochs=3),
        device=torch.device("cpu"),
        logger=make_logger(),
        output_dir=str(tmp_path),
    )

    train_results = [(0.8, 0.7), (0.7, 0.75), (0.6, 0.74)]
    val_results = [
        (0.40, 0.80, {"precision": 0.80, "recall": 0.80, "f1": 0.80, "precision_macro": 0.50, "recall_macro": 0.50, "f1_macro": 0.50}),
        (0.55, 0.78, {"precision": 0.78, "recall": 0.78, "f1": 0.78, "precision_macro": 0.65, "recall_macro": 0.65, "f1_macro": 0.65}),
        (0.35, 0.79, {"precision": 0.79, "recall": 0.79, "f1": 0.79, "precision_macro": 0.60, "recall_macro": 0.60, "f1_macro": 0.60}),
    ]

    trainer.train_epoch = lambda: train_results.pop(0)
    trainer.validate = lambda: val_results.pop(0)

    saved_checkpoints = []
    trainer._save_checkpoint = lambda filename: saved_checkpoints.append(filename)

    history = trainer.train()

    assert history["val_f1_macro"] == [0.50, 0.65, 0.60]
    assert trainer.best_epoch == 2
    assert trainer.best_monitor_value == 0.65
    assert trainer.best_val_loss == 0.35
    assert trainer.best_val_acc == 0.80
    assert saved_checkpoints.count("best_model.pth") == 2
