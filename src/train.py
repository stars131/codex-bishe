"""
Training utilities for network attack detection models.
"""
import argparse
import copy
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_multi_source_loaders, get_class_weights
from src.models.fusion_net import create_model
from src.models.losses import create_loss_function
from src.utils.helpers import load_config, set_seed


def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    log_path = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class WarmupScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Any,
        warmup_start_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        if self.current_epoch <= self.warmup_epochs:
            warmup_factor = self.current_epoch / max(self.warmup_epochs, 1)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
        elif self.base_scheduler is not None:
            self.base_scheduler.step()

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_lr": self.warmup_start_lr,
            "base_lrs": self.base_lrs,
            "base_scheduler_state_dict": self.base_scheduler.state_dict() if self.base_scheduler else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_epoch = state_dict.get("current_epoch", self.current_epoch)
        self.warmup_epochs = state_dict.get("warmup_epochs", self.warmup_epochs)
        self.warmup_start_lr = state_dict.get("warmup_start_lr", self.warmup_start_lr)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)
        base_state = state_dict.get("base_scheduler_state_dict")
        if self.base_scheduler is not None and base_state is not None:
            self.base_scheduler.load_state_dict(base_state)


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
    **kwargs,
):
    if scheduler_type == "cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "cosine_warm_restarts":
        base_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "step":
        base_scheduler = StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_type == "reduce_on_plateau":
        base_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("gamma", 0.1),
            patience=kwargs.get("patience", 10),
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", kwargs.get("learning_rate", 0.001) * 10),
            epochs=epochs,
            steps_per_epoch=kwargs.get("steps_per_epoch", 100),
            pct_start=kwargs.get("pct_start", 0.3),
        )
    else:
        base_scheduler = None

    if warmup_epochs > 0 and base_scheduler is not None:
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
    return base_scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger: logging.Logger,
        output_dir: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 100)
        self.gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        requested_amp = train_config.get("mixed_precision", False)
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.use_amp = requested_amp and self.amp_device_type == "cuda"
        if requested_amp and not self.use_amp:
            self.logger.warning("Mixed precision requested but CUDA is unavailable; AMP disabled.")
        self.max_grad_norm = train_config.get("gradient_clip", {}).get("max_norm", 1.0)

        es_config = train_config.get("early_stopping", {})
        self.early_stopping_enabled = es_config.get("enabled", True)
        self.early_stopping_patience = es_config.get("patience", 15)
        self.early_stopping_min_delta = es_config.get("min_delta", 0.001)
        self.monitor_metric = es_config.get("monitor", "val_loss")
        self.monitor_mode = es_config.get("mode") or ("min" if "loss" in self.monitor_metric else "max")

        self._init_criterion()
        self._init_optimizer()
        self._init_scheduler()

        self.scaler = GradScaler(device="cuda") if self.use_amp else None

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_monitor_value = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_precision_macro": [],
            "val_recall_macro": [],
            "val_f1_macro": [],
            "learning_rate": [],
        }

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
        except ImportError:
            self.writer = None
            self.logger.warning("TensorBoard not available")

    def _init_criterion(self):
        loss_config = self.config.get("training", {}).get("loss", {})
        loss_type = loss_config.get("type", "cross_entropy")
        num_classes = self.config.get("model", {}).get("num_classes", 2)
        samples_per_class = None

        class_weights = loss_config.get("class_weights")
        labels = self._extract_dataset_labels(self.train_loader.dataset)
        if class_weights == "auto" and labels is not None:
            class_weights = get_class_weights(labels, num_classes=num_classes).tolist()
        if loss_type == "class_balanced" and labels is not None:
            samples_per_class = np.bincount(labels, minlength=num_classes).tolist()

        self.criterion = create_loss_function(
            loss_type=loss_type,
            num_classes=num_classes,
            class_weights=class_weights,
            samples_per_class=samples_per_class,
            gamma=loss_config.get("focal_gamma", 2.0),
            gamma_neg=loss_config.get("gamma_neg", 4.0),
            gamma_pos=loss_config.get("gamma_pos", 1.0),
            use_softmax=loss_config.get("use_softmax", False),
            smoothing=loss_config.get("smoothing", 0.1),
            beta=loss_config.get("beta", 0.9999),
            class_balanced_loss_type=loss_config.get("class_balanced_loss_type", "focal"),
            label_smoothing=loss_config.get("label_smoothing", 0.0),
        )
        self.logger.info(f"Loss function: {loss_type}")
        if class_weights is not None:
            self.logger.info(f"Class weights: {class_weights}")
        if samples_per_class is not None:
            self.logger.info(f"Samples per class: {samples_per_class}")

    @staticmethod
    def _extract_dataset_labels(dataset) -> Optional[np.ndarray]:
        if hasattr(dataset, "labels"):
            labels = dataset.labels
        elif hasattr(dataset, "tensors") and dataset.tensors:
            labels = dataset.tensors[-1]
        else:
            return None

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return np.asarray(labels)

    def _init_optimizer(self):
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "adamw").lower()
        params = self.model.parameters()

        if opt_type == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=opt_config.get("learning_rate", 0.001),
                weight_decay=opt_config.get("weight_decay", 0.0001),
                betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            )
        elif opt_type == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=opt_config.get("learning_rate", 0.001),
                weight_decay=opt_config.get("weight_decay", 0.0001),
                betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            )
        elif opt_type == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=opt_config.get("learning_rate", 0.01),
                weight_decay=opt_config.get("weight_decay", 0.0001),
                momentum=opt_config.get("momentum", 0.9),
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        self.logger.info(f"Optimizer: {opt_type}, LR: {opt_config.get('learning_rate', 0.001)}")

    def _init_scheduler(self):
        sched_config = self.config.get("training", {}).get("scheduler", {})
        sched_type = sched_config.get("type", "cosine")
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=sched_type,
            epochs=self.epochs,
            warmup_epochs=sched_config.get("warmup_epochs", 5),
            min_lr=sched_config.get("min_lr", 1e-6),
            step_size=sched_config.get("step_size", 30),
            gamma=sched_config.get("gamma", 0.1),
            steps_per_epoch=len(self.train_loader),
            learning_rate=self.config.get("training", {}).get("optimizer", {}).get("learning_rate", 0.001),
        )
        self.logger.info(f"Scheduler: {sched_type}")

    def _prepare_batch(self, batch):
        if len(batch) >= 3:
            *source_features, labels = batch
            return [source.to(self.device) for source in source_features], labels.to(self.device), True
        features, labels = batch
        return features.to(self.device), labels.to(self.device), False

    def _forward_batch(self, batch):
        inputs, labels, is_multi_source = self._prepare_batch(batch)
        if is_multi_source:
            outputs = self.model(*inputs)
        else:
            outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs, labels

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                outputs, labels = self._forward_batch(batch)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            outputs, labels = self._forward_batch(batch)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import f1_score, precision_score, recall_score

        metrics = {
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }
        return total_loss / total, correct / total, metrics

    def _get_monitor_value(self, val_loss: float, val_acc: float, val_metrics: Dict[str, float]) -> float:
        monitor_candidates = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_accuracy": val_acc,
        }
        for key, value in val_metrics.items():
            monitor_candidates[f"val_{key}"] = value

        if self.monitor_metric not in monitor_candidates:
            available = ", ".join(sorted(monitor_candidates.keys()))
            raise ValueError(f"Unknown monitor metric: {self.monitor_metric}. Available metrics: {available}")
        return float(monitor_candidates[self.monitor_metric])

    def _is_monitor_improved(self, current_value: float) -> bool:
        if self.monitor_mode == "min":
            return current_value < self.best_monitor_value - self.early_stopping_min_delta
        return current_value > self.best_monitor_value + self.early_stopping_min_delta

    def train(self) -> Dict:
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision_macro"].append(val_metrics["precision_macro"])
            self.history["val_recall_macro"].append(val_metrics["recall_macro"])
            self.history["val_f1_macro"].append(val_metrics["f1_macro"])
            self.history["learning_rate"].append(current_lr)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("Metrics/val_precision", val_metrics["precision"], epoch)
                self.writer.add_scalar("Metrics/val_recall", val_metrics["recall"], epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)
                self.writer.add_scalar("Metrics/val_f1", val_metrics["f1"], epoch)
                self.writer.add_scalar("Metrics/val_f1_macro", val_metrics["f1_macro"], epoch)

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | Macro F1: {val_metrics['f1_macro']:.4f} | LR: {current_lr:.6f} | "
                f"Time: {time.time() - epoch_start:.1f}s"
            )

            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.best_val_acc = max(self.best_val_acc, val_acc)
            monitor_value = self._get_monitor_value(val_loss, val_acc, val_metrics)
            is_best = self._is_monitor_improved(monitor_value)
            if is_best:
                self.best_monitor_value = monitor_value
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pth")
                self.logger.info(
                    f"  New best model saved! {self.monitor_metric}: {monitor_value:.4f}"
                )
            else:
                self.epochs_without_improvement += 1

            save_every = self.config.get("training", {}).get("checkpoint", {}).get("save_every", 10)
            if epoch % save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

            if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        self._save_checkpoint("last_model.pth")
        if self.writer is not None:
            self.writer.close()

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.4f}")
        self.logger.info(
            f"Best checkpoint metric ({self.monitor_metric}, {self.monitor_mode}): "
            f"{self.best_monitor_value:.4f} at epoch {self.best_epoch}"
        )
        return self.history

    def _save_checkpoint(self, filename: str):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_monitor_value": self.best_monitor_value,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": self.config,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        path = os.path.join(self.output_dir, "checkpoints", filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.best_monitor_value = checkpoint.get("best_monitor_value", self.best_monitor_value)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        self.history = checkpoint.get("history", self.history)
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in data_loader:
            outputs, labels = self._forward_batch(batch)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import f1_score, precision_score, recall_score

        metrics = {
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        }
        return total_loss / total, correct / total, metrics


class AblationStudy:
    def __init__(
        self,
        base_config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: str,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.results = {}

    def run_experiment(self, experiment_name: str, model_config: Dict, epochs: int = 50) -> Dict:
        print(f"\n{'=' * 50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'=' * 50}")

        config = copy.deepcopy(self.base_config)
        config["training"]["epochs"] = epochs
        config["model"].update({k: v for k, v in model_config.items() if k in ["type", "source_dims", "source1_dim", "source2_dim", "num_classes"]})

        factory_config = dict(config["model"].get("architecture", {}))
        factory_config["fusion_type"] = model_config.get("fusion_type", factory_config.get("fusion_type", "attention"))
        factory_config["encoder_type"] = model_config.get("encoder_type", factory_config.get("encoder_type", "mlp"))
        factory_config["source_dims"] = config["model"].get("source_dims", [config["model"]["source1_dim"], config["model"]["source2_dim"]])
        factory_config["decision_fusion"] = copy.deepcopy(config["model"].get("decision_fusion", {}))
        factory_config["agentic_mode"] = copy.deepcopy(config["model"].get("agentic_mode", {}))

        model = create_model(
            model_type=config["model"].get("type", "fusion_net"),
            traffic_dim=config["model"]["source1_dim"],
            log_dim=config["model"]["source2_dim"],
            num_classes=config["model"]["num_classes"],
            config=factory_config,
        )

        exp_output_dir = os.path.join(self.output_dir, experiment_name)
        logger = setup_logger(exp_output_dir, experiment_name)

        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=config,
            device=self.device,
            logger=logger,
            output_dir=exp_output_dir,
        )

        history = trainer.train()
        test_loss, test_acc, test_metrics = trainer.evaluate(self.test_loader)

        result = {
            "name": experiment_name,
            "config": model_config,
            "best_val_loss": trainer.best_val_loss,
            "best_val_acc": trainer.best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_metrics": test_metrics,
            "history": history,
        }
        self.results[experiment_name] = result
        return result

    def compare_fusion_methods(self):
        source_dims = self.base_config.get("model", {}).get("source_dims", [])
        fusion_types = ["attention", "multi_head", "gated", "cross", "concat"]
        if len(source_dims) > 2:
            fusion_types = [ft for ft in fusion_types if ft != "cross"]

        for fusion_type in fusion_types:
            self.run_experiment(f"fusion_{fusion_type}", {"fusion_type": fusion_type})

    def compare_encoders(self):
        for encoder_type in ["mlp", "cnn", "lstm", "transformer"]:
            self.run_experiment(f"encoder_{encoder_type}", {"encoder_type": encoder_type})

    def get_summary(self) -> str:
        summary = "\n" + "=" * 60 + "\n"
        summary += "Ablation Study Results\n"
        summary += "=" * 60 + "\n"
        for name, result in self.results.items():
            summary += f"\n{name}:\n"
            summary += f"  Best Val Acc: {result['best_val_acc']:.4f}\n"
            summary += f"  Test Acc: {result['test_acc']:.4f}\n"
            summary += f"  Test F1: {result['test_metrics']['f1']:.4f}\n"
        return summary


def _discover_source_indices(data: Dict) -> List[int]:
    return sorted({
        int(match.group(1))
        for key in data.keys()
        for match in [re.match(r"^s(\d+)_train$", key)]
        if match is not None
    })


def _build_multi_source_data_dict(data: Dict) -> Tuple[Dict, List[int]]:
    source_indices = _discover_source_indices(data)
    if len(source_indices) < 2:
        raise ValueError("multi_source_data.pkl must contain at least s1_* and s2_*")

    data_dict = {
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
    }
    for idx in source_indices:
        data_dict[f"X{idx}_train"] = data[f"s{idx}_train"]
        data_dict[f"X{idx}_val"] = data[f"s{idx}_val"]
        data_dict[f"X{idx}_test"] = data[f"s{idx}_test"]
    return data_dict, source_indices


def _infer_source_dims(data: Dict, source_indices: List[int]) -> List[int]:
    return [data[f"s{idx}_train"].shape[1] for idx in source_indices]


def main():
    parser = argparse.ArgumentParser(description="Train network attack detection model")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    logger = setup_logger(output_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")

    data_path = os.path.join(args.data_dir, "multi_source_data.pkl")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please run data preprocessing first: python main.py --mode preprocess")
        return

    import pickle

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data_dict, source_indices = _build_multi_source_data_dict(data)
    source_dims = _infer_source_dims(data, source_indices)

    config["model"]["source_dims"] = source_dims
    config["model"]["source1_dim"] = source_dims[0]
    config["model"]["source2_dim"] = source_dims[1]
    config["model"]["num_classes"] = data.get("num_classes", len(np.unique(data["y_train"])))

    loader_config = config.get("data", {}).get("loader", {})
    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=config["training"]["batch_size"],
        num_workers=loader_config.get("num_workers", 4),
        pin_memory=loader_config.get("pin_memory", device.type == "cuda"),
        use_weighted_sampler=loader_config.get("use_weighted_sampler", False),
        augment_train=loader_config.get("augment_train", False),
    )

    if args.ablation:
        ablation = AblationStudy(
            base_config=config,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            device=device,
            output_dir=os.path.join(output_dir, "ablation"),
        )
        ablation.compare_fusion_methods()
        ablation.compare_encoders()
        logger.info(ablation.get_summary())

        import json

        with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
            serializable_results = {}
            for name, result in ablation.results.items():
                serializable_results[name] = {
                    "name": result["name"],
                    "best_val_acc": float(result["best_val_acc"]),
                    "test_acc": float(result["test_acc"]),
                    "test_metrics": {k: float(v) for k, v in result["test_metrics"].items()},
                }
            json.dump(serializable_results, f, indent=2)
        return

    model_config = dict(config.get("model", {}).get("architecture", {}))
    model_config["source_dims"] = config["model"]["source_dims"]
    model_config["fusion_type"] = config.get("model", {}).get("fusion", {}).get("method", "attention")
    model_config["decision_fusion"] = copy.deepcopy(config.get("model", {}).get("decision_fusion", {}))
    model_config["agentic_mode"] = copy.deepcopy(config.get("model", {}).get("agentic_mode", {}))

    model = create_model(
        model_type=config["model"].get("type", "fusion_net"),
        traffic_dim=config["model"]["source1_dim"],
        log_dim=config["model"]["source2_dim"],
        num_classes=config["model"]["num_classes"],
        config=model_config,
    )

    logger.info(f"Model: {config['model'].get('type', 'fusion_net')}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        device=device,
        logger=logger,
        output_dir=output_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_metrics = trainer.evaluate(loaders["test"])
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
