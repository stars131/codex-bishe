"""
训练监控和回调模块

提供实时训练监控、TensorBoard日志记录和训练回调功能。
"""
import os
import json
import time
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Callable
import numpy as np

try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GPUMonitor:
    """
    GPU资源监控器

    监控GPU内存使用、利用率等信息。
    """

    def __init__(self):
        self.has_gpu = HAS_TORCH and torch.cuda.is_available()
        self.history = {
            'memory_allocated': [],
            'memory_reserved': [],
            'memory_percent': [],
            'gpu_utilization': []
        }

    def get_memory_info(self) -> Dict:
        """获取当前GPU内存信息"""
        if not self.has_gpu:
            return {'allocated': 0, 'reserved': 0, 'total': 0, 'percent': 0}

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        percent = (allocated / total) * 100 if total > 0 else 0

        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'percent': percent
        }

    def log_memory(self):
        """记录当前GPU内存使用"""
        info = self.get_memory_info()
        self.history['memory_allocated'].append(info['allocated'])
        self.history['memory_reserved'].append(info['reserved'])
        self.history['memory_percent'].append(info['percent'])

    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.history['memory_allocated']:
            return {}

        return {
            'peak_allocated_gb': max(self.history['memory_allocated']),
            'avg_allocated_gb': np.mean(self.history['memory_allocated']),
            'peak_percent': max(self.history['memory_percent']),
            'avg_percent': np.mean(self.history['memory_percent'])
        }

    def clear_cache(self):
        """清理GPU缓存"""
        if self.has_gpu:
            torch.cuda.empty_cache()

    @staticmethod
    def get_device_info() -> Dict:
        """获取GPU设备信息"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {'available': False}

        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }


class PerformanceTracker:
    """
    性能追踪器

    追踪训练过程中的时间、吞吐量等性能指标。
    """

    def __init__(self):
        self.epoch_times = []
        self.batch_times = []
        self.throughputs = []
        self.epoch_start = None
        self.batch_start = None

    def start_epoch(self):
        """开始计时一个epoch"""
        self.epoch_start = time.time()
        self.batch_times = []

    def end_epoch(self, num_samples: int = None) -> float:
        """
        结束一个epoch计时

        Args:
            num_samples: 本epoch处理的样本数

        Returns:
            epoch耗时（秒）
        """
        if self.epoch_start is None:
            return 0

        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)

        if num_samples:
            throughput = num_samples / elapsed
            self.throughputs.append(throughput)

        return elapsed

    def start_batch(self):
        """开始计时一个batch"""
        self.batch_start = time.time()

    def end_batch(self) -> float:
        """结束一个batch计时"""
        if self.batch_start is None:
            return 0

        elapsed = time.time() - self.batch_start
        self.batch_times.append(elapsed)
        return elapsed

    def get_summary(self) -> Dict:
        """获取性能摘要"""
        summary = {}

        if self.epoch_times:
            summary['total_time'] = sum(self.epoch_times)
            summary['avg_epoch_time'] = np.mean(self.epoch_times)
            summary['min_epoch_time'] = min(self.epoch_times)
            summary['max_epoch_time'] = max(self.epoch_times)

        if self.throughputs:
            summary['avg_throughput'] = np.mean(self.throughputs)
            summary['peak_throughput'] = max(self.throughputs)

        if self.batch_times:
            summary['avg_batch_time'] = np.mean(self.batch_times)

        return summary

    def estimate_remaining_time(self, current_epoch: int, total_epochs: int) -> float:
        """估计剩余训练时间"""
        if not self.epoch_times:
            return 0

        avg_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        return avg_time * remaining_epochs


class TrainingHistory:
    """
    训练历史记录器

    记录和管理训练过程中的所有指标。
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': [],
            'metrics': {}
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.start_time = None

    def start_training(self):
        """开始训练计时"""
        self.start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float = None,
        val_acc: float = None,
        lr: float = None,
        metrics: Dict[str, float] = None,
        epoch_time: float = None
    ):
        """记录一个epoch的数据"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        if lr is not None:
            self.history['learning_rate'].append(lr)
        if epoch_time is not None:
            self.history['epoch_time'].append(epoch_time)

        # 记录额外指标
        if metrics:
            for key, value in metrics.items():
                if key not in self.history['metrics']:
                    self.history['metrics'][key] = []
                self.history['metrics'][key].append(value)

        # 更新最佳记录
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

    def get_summary(self) -> Dict:
        """获取训练摘要"""
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            'total_epochs': len(self.history['train_loss']),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'total_training_time': total_time,
            'avg_epoch_time': np.mean(self.history['epoch_time']) if self.history['epoch_time'] else None
        }

    def save(self, path: str):
        """保存训练历史"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'history': self.history,
            'summary': self.get_summary()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"训练历史已保存: {path}")

    @staticmethod
    def load(path: str) -> 'TrainingHistory':
        """加载训练历史"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        history = TrainingHistory()
        history.history = data['history']
        return history


class TensorBoardLogger:
    """
    TensorBoard 日志记录器

    记录训练过程到TensorBoard。
    """

    def __init__(self, log_dir: str, experiment_name: str = None):
        if not HAS_TORCH:
            raise ImportError("需要安装 PyTorch 和 TensorBoard")

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard 日志目录: {self.log_dir}")

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量"""
        for name, value in values.items():
            self.writer.add_scalar(f"{tag}/{name}", value, step)

    def log_scalar(self, tag: str, value: float, step: int):
        """记录单个标量"""
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_tensor):
        """记录模型结构图"""
        self.writer.add_graph(model, input_tensor)

    def log_image(self, tag: str, image, step: int):
        """记录图片"""
        self.writer.add_image(tag, image, step)

    def log_figure(self, tag: str, figure, step: int):
        """记录 matplotlib 图表"""
        self.writer.add_figure(tag, figure, step)

    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        self.writer.add_text(tag, text, step)

    def log_hyperparams(self, hparams: Dict, metrics: Dict):
        """记录超参数"""
        self.writer.add_hparams(hparams, metrics)

    def log_embedding(self, features, metadata=None, label_img=None, tag='embedding', step=0):
        """记录嵌入向量"""
        self.writer.add_embedding(features, metadata=metadata, label_img=label_img, tag=tag, global_step=step)

    def close(self):
        """关闭日志记录器"""
        self.writer.close()


class EarlyStopping:
    """
    早停机制

    当验证指标不再改善时停止训练。
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: 等待改善的epoch数
            min_delta: 最小改善幅度
            mode: 'min' 表示越小越好，'max' 表示越大越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前epoch的指标值
            epoch: 当前epoch

        Returns:
            是否应该早停
        """
        if self.mode == 'min':
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f"  ✓ 验证指标改善: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ✗ 验证指标未改善 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n早停触发! 最佳 epoch: {self.best_epoch}, 最佳分数: {self.best_score:.6f}")

        return self.early_stop


class ModelCheckpoint:
    """
    模型检查点管理器

    保存最佳模型和定期检查点。
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best: bool = True,
        save_last: bool = True,
        save_every: int = None,
        verbose: bool = True
    ):
        """
        Args:
            save_dir: 保存目录
            monitor: 监控的指标名
            mode: 'min' 或 'max'
            save_best: 是否保存最佳模型
            save_last: 是否保存最后一个模型
            save_every: 每N个epoch保存一次
            verbose: 是否打印信息
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_every = save_every
        self.verbose = verbose

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0

        os.makedirs(save_dir, exist_ok=True)

    def __call__(
        self,
        model,
        optimizer,
        epoch: int,
        score: float,
        extra_info: Dict = None
    ):
        """
        检查并保存模型

        Args:
            model: PyTorch 模型
            optimizer: 优化器
            epoch: 当前epoch
            score: 监控指标的值
            extra_info: 额外信息
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            self.monitor: score
        }
        if extra_info:
            checkpoint.update(extra_info)

        # 保存最佳模型
        if self.save_best:
            is_best = (self.mode == 'min' and score < self.best_score) or \
                      (self.mode == 'max' and score > self.best_score)
            if is_best:
                self.best_score = score
                self.best_epoch = epoch
                best_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                if self.verbose:
                    print(f"  ✓ 保存最佳模型: {best_path}")

        # 保存最后一个模型
        if self.save_last:
            last_path = os.path.join(self.save_dir, 'last_model.pth')
            torch.save(checkpoint, last_path)

        # 定期保存
        if self.save_every and epoch % self.save_every == 0:
            periodic_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)
            if self.verbose:
                print(f"  ✓ 保存检查点: {periodic_path}")


class ProgressBar:
    """
    训练进度条

    显示训练进度和实时指标。
    """

    def __init__(self, total: int, desc: str = "Training", ncols: int = 100):
        self.total = total
        self.desc = desc
        self.ncols = ncols
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1, metrics: Dict[str, float] = None):
        """更新进度"""
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        # 进度条
        percent = self.current / self.total
        bar_length = 30
        filled = int(bar_length * percent)
        bar = '█' * filled + '░' * (bar_length - filled)

        # 指标字符串
        metrics_str = ""
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        # 打印
        eta = (self.total - self.current) / rate if rate > 0 else 0
        print(f"\r{self.desc}: |{bar}| {self.current}/{self.total} "
              f"[{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s] {metrics_str}", end='')

        if self.current >= self.total:
            print()

    def reset(self):
        """重置进度条"""
        self.current = 0
        self.start_time = time.time()


class TrainingMonitor:
    """
    训练监控器

    整合所有监控功能的高级接口。
    """

    def __init__(
        self,
        experiment_name: str = None,
        log_dir: str = "outputs/logs",
        checkpoint_dir: str = "outputs/checkpoints",
        use_tensorboard: bool = True,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        verbose: bool = True
    ):
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verbose = verbose

        # 训练历史
        self.history = TrainingHistory()

        # TensorBoard
        self.tb_logger = None
        if use_tensorboard and HAS_TORCH:
            self.tb_logger = TensorBoardLogger(log_dir, self.experiment_name)

        # 早停
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=verbose
        )

        # 检查点
        self.checkpoint = None
        if save_best and HAS_TORCH:
            self.checkpoint = ModelCheckpoint(
                save_dir=os.path.join(checkpoint_dir, self.experiment_name),
                verbose=verbose
            )

        # 结果保存目录
        self.results_dir = os.path.join("outputs/results", self.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)

    def on_train_begin(self, config: Dict = None):
        """训练开始时调用"""
        self.history.start_training()
        if self.verbose:
            print("=" * 60)
            print(f"实验名称: {self.experiment_name}")
            print("=" * 60)

        if config and self.tb_logger:
            self.tb_logger.log_text("config", json.dumps(config, indent=2), 0)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float = None,
        val_acc: float = None,
        lr: float = None,
        metrics: Dict[str, float] = None,
        model=None,
        optimizer=None
    ) -> bool:
        """
        每个epoch结束时调用

        Returns:
            是否应该早停
        """
        # 记录历史
        self.history.log_epoch(
            epoch, train_loss, val_loss, train_acc, val_acc, lr, metrics
        )

        # TensorBoard 记录
        if self.tb_logger:
            self.tb_logger.log_scalars("Loss", {
                "train": train_loss,
                "val": val_loss
            }, epoch)

            if train_acc is not None:
                self.tb_logger.log_scalars("Accuracy", {
                    "train": train_acc,
                    "val": val_acc
                }, epoch)

            if lr is not None:
                self.tb_logger.log_scalar("Learning_Rate", lr, epoch)

            if metrics:
                for key, value in metrics.items():
                    self.tb_logger.log_scalar(f"Metrics/{key}", value, epoch)

        # 保存检查点
        if self.checkpoint and model and optimizer:
            self.checkpoint(model, optimizer, epoch, val_loss, {
                'val_acc': val_acc,
                'metrics': metrics
            })

        # 打印进度
        if self.verbose:
            metrics_str = ""
            if metrics:
                metrics_str = " | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
                  f"{metrics_str}")

        # 早停检查
        return self.early_stopping(val_loss, epoch)

    def on_train_end(self):
        """训练结束时调用"""
        # 保存训练历史
        history_path = os.path.join(self.results_dir, "training_history.pkl")
        self.history.save(history_path)

        # 打印摘要
        summary = self.history.get_summary()
        if self.verbose:
            print("\n" + "=" * 60)
            print("训练完成!")
            print("=" * 60)
            print(f"总训练轮数: {summary['total_epochs']}")
            print(f"最佳 Epoch: {summary['best_epoch']}")
            print(f"最佳验证损失: {summary['best_val_loss']:.6f}")
            print(f"最佳验证准确率: {summary['best_val_acc']:.6f}")
            print(f"总训练时间: {summary['total_training_time']:.1f}s")
            print("=" * 60)

        # 关闭 TensorBoard
        if self.tb_logger:
            self.tb_logger.close()

        return summary

    def save_model_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        class_names: List[str] = None,
        attention_weights: np.ndarray = None,
        metrics: Dict[str, float] = None
    ):
        """保存模型评估结果"""
        from sklearn.metrics import confusion_matrix, classification_report

        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'class_names': class_names,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
            'metrics': metrics,
            'attention_weights': attention_weights
        }

        results_path = os.path.join(self.results_dir, "model_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        if self.verbose:
            print(f"模型结果已保存: {results_path}")
