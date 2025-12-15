"""
Utility functions for training, evaluation, logging, and model management.

Author: Network Attack Detection Project
"""
import os
import sys
import yaml
import json
import pickle
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)


# ============================================
# 配置管理
# ============================================

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """递归合并配置字典"""
    result = base_config.copy()
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# ============================================
# 随机种子
# ============================================

def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================
# 检查点管理
# ============================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Any = None,
    scaler: Any = None,
    metrics: Dict = None,
    config: Dict = None
) -> None:
    """保存模型检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    scaler: Any = None,
    device: torch.device = None
) -> Dict:
    """加载模型检查点"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """获取最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') or f.endswith('.pt')
    ]

    if not checkpoints:
        return None

    # 按修改时间排序
    checkpoints.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )

    return os.path.join(checkpoint_dir, checkpoints[0])


# ============================================
# 评估指标
# ============================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    class_names: List[str] = None,
    average: str = 'weighted'
) -> Dict[str, Any]:
    """
    计算完整的评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        class_names: 类别名称（可选）
        average: 多分类平均方式

    Returns:
        包含各种指标的字典
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # AUC-ROC
    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2

            if num_classes == 2:
                prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics['auc_roc'] = float(roc_auc_score(y_true, prob))
            else:
                metrics['auc_roc'] = float(roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average=average
                ))
        except (ValueError, IndexError):
            metrics['auc_roc'] = None

    # 每类指标
    if class_names is not None:
        per_class = {}
        for i, name in enumerate(class_names):
            mask = y_true == i
            if mask.sum() > 0:
                per_class[name] = {
                    'precision': float(precision_score(y_true == i, y_pred == i, zero_division=0)),
                    'recall': float(recall_score(y_true == i, y_pred == i, zero_division=0)),
                    'f1': float(f1_score(y_true == i, y_pred == i, zero_division=0)),
                    'support': int(mask.sum())
                }
        metrics['per_class'] = per_class

    return metrics


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = None
) -> Dict[str, Any]:
    """计算ROC曲线数据"""
    if num_classes is None:
        num_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2

    result = {}

    if num_classes == 2:
        prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        fpr, tpr, thresholds = roc_curve(y_true, prob)
        result['fpr'] = fpr.tolist()
        result['tpr'] = tpr.tolist()
        result['thresholds'] = thresholds.tolist()
    else:
        for i in range(num_classes):
            fpr, tpr, thresholds = roc_curve(y_true == i, y_prob[:, i])
            result[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }

    return result


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = None
) -> Dict[str, Any]:
    """计算PR曲线数据"""
    if num_classes is None:
        num_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2

    result = {}

    if num_classes == 2:
        prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        precision, recall, thresholds = precision_recall_curve(y_true, prob)
        result['precision'] = precision.tolist()
        result['recall'] = recall.tolist()
        result['thresholds'] = thresholds.tolist()
    else:
        for i in range(num_classes):
            precision, recall, thresholds = precision_recall_curve(
                y_true == i, y_prob[:, i]
            )
            result[f'class_{i}'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            }

    return result


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation Results") -> None:
    """格式化打印评估指标"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    for name, value in metrics.items():
        if name in ['confusion_matrix', 'per_class', 'roc_curve', 'pr_curve']:
            continue
        if value is not None:
            if isinstance(value, float):
                print(f"  {name:20s}: {value:.4f}")
            else:
                print(f"  {name:20s}: {value}")

    if 'per_class' in metrics:
        print("\n  Per-class metrics:")
        print("  " + "-" * 50)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"    {class_name}:")
            for m_name, m_value in class_metrics.items():
                if isinstance(m_value, float):
                    print(f"      {m_name}: {m_value:.4f}")
                else:
                    print(f"      {m_name}: {m_value}")

    print("=" * 60)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> str:
    """生成分类报告"""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


# ============================================
# 日志工具
# ============================================

def setup_logger(
    name: str,
    log_dir: str,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # 清除现有handlers

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件handler
    log_file = os.path.join(
        log_dir,
        f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class TensorBoardLogger:
    """TensorBoard日志封装"""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            self.writer = None
            self.enabled = False
            print("Warning: TensorBoard not available")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)

    def log_figure(self, tag: str, figure, step: int):
        if self.enabled:
            self.writer.add_figure(tag, figure, step)

    def log_text(self, tag: str, text: str, step: int):
        if self.enabled:
            self.writer.add_text(tag, text, step)

    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        if self.enabled:
            self.writer.add_graph(model, input_tensor)

    def close(self):
        if self.enabled:
            self.writer.close()


# ============================================
# 数据工具
# ============================================

def save_results(
    results: Dict,
    path: str,
    format: str = 'json'
) -> None:
    """保存实验结果"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if format == 'json':
        # 转换numpy数组为列表
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)

    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(results, f)

    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(path: str) -> Dict:
    """加载实验结果"""
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {path}")


# ============================================
# 模型工具
# ============================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """计算模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size(model: nn.Module) -> float:
    """获取模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """冻结指定层"""
    for name, param in model.named_parameters():
        if any(ln in name for ln in layer_names):
            param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: List[str] = None) -> None:
    """解冻指定层（如果layer_names为None则解冻所有层）"""
    for name, param in model.named_parameters():
        if layer_names is None or any(ln in name for ln in layer_names):
            param.requires_grad = True


def get_device(device_str: str = 'auto') -> torch.device:
    """获取计算设备"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def move_to_device(data: Any, device: torch.device) -> Any:
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


# ============================================
# 时间工具
# ============================================

class Timer:
    """计时器"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        self.start_time = datetime.now()

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = (datetime.now() - self.start_time).total_seconds()
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ============================================
# 进度工具
# ============================================

class ProgressBar:
    """简单的进度条"""

    def __init__(self, total: int, prefix: str = '', length: int = 50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0

    def update(self, n: int = 1):
        self.current += n
        percent = self.current / self.total
        filled = int(self.length * percent)
        bar = '█' * filled + '-' * (self.length - filled)
        print(f'\r{self.prefix} |{bar}| {percent*100:.1f}%', end='', flush=True)
        if self.current >= self.total:
            print()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.current < self.total:
            print()
