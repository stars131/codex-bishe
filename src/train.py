"""
Advanced training script for network attack detection.
Supports mixed precision, gradient accumulation, learning rate warmup, and more.

Author: Network Attack Detection Project
"""
import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    StepLR, ReduceLROnPlateau, OneCycleLR
)

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config, set_seed, save_checkpoint, load_checkpoint
from src.models.fusion_net import FusionNet, SingleSourceNet, create_model
from src.models.losses import create_loss_function, FocalLoss, ContrastiveLoss, CenterLoss
from src.data.dataset import (
    create_data_loaders, create_multi_source_loaders,
    get_class_weights
)


# ============================================
# 日志配置
# ============================================

def setup_logger(log_dir: str, name: str = 'train') -> logging.Logger:
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除现有handlers
    logger.handlers = []

    # 文件handler
    fh = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    fh.setLevel(logging.INFO)

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================
# 学习率调度器
# ============================================

class WarmupScheduler:
    """学习率预热调度器"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Any,
        warmup_start_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr

        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # 线性预热
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
        else:
            # 使用基础调度器
            self.base_scheduler.step()

    def get_last_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
    **kwargs
) -> Any:
    """创建学习率调度器"""

    if scheduler_type == 'cosine':
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cosine_warm_restarts':
        base_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        base_scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'reduce_on_plateau':
        base_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('gamma', 0.1),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', kwargs.get('learning_rate', 0.001) * 10),
            epochs=epochs,
            steps_per_epoch=kwargs.get('steps_per_epoch', 100),
            pct_start=kwargs.get('pct_start', 0.3)
        )
    else:
        base_scheduler = None

    # 添加预热
    if warmup_epochs > 0 and base_scheduler is not None:
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)

    return base_scheduler


# ============================================
# 训练器类
# ============================================

class Trainer:
    """
    高级训练器

    支持:
    - 混合精度训练
    - 梯度累积
    - 学习率预热
    - 早停
    - 模型检查点
    - TensorBoard日志
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger: logging.Logger,
        output_dir: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

        # 训练配置
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        self.use_amp = train_config.get('mixed_precision', False)
        self.max_grad_norm = train_config.get('gradient_clip', {}).get('max_norm', 1.0)

        # 早停配置
        es_config = train_config.get('early_stopping', {})
        self.early_stopping_enabled = es_config.get('enabled', True)
        self.early_stopping_patience = es_config.get('patience', 15)
        self.early_stopping_min_delta = es_config.get('min_delta', 0.001)

        # 初始化损失函数
        self._init_criterion()

        # 初始化优化器
        self._init_optimizer()

        # 初始化调度器
        self._init_scheduler()

        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.epochs_without_improvement = 0

        # 历史记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        except ImportError:
            self.writer = None
            self.logger.warning("TensorBoard not available")

    def _init_criterion(self):
        """初始化损失函数"""
        loss_config = self.config.get('training', {}).get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy')

        # 获取类别权重
        class_weights = loss_config.get('class_weights')
        if class_weights == 'auto':
            # 从数据计算权重
            if hasattr(self.train_loader.dataset, 'labels'):
                labels = self.train_loader.dataset.labels.numpy()
                class_weights = get_class_weights(labels).tolist()

        self.criterion = create_loss_function(
            loss_type=loss_type,
            num_classes=self.config.get('model', {}).get('num_classes', 2),
            class_weights=class_weights,
            gamma=loss_config.get('focal_gamma', 2.0),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
        self.logger.info(f"Loss function: {loss_type}")

    def _init_optimizer(self):
        """初始化优化器"""
        opt_config = self.config.get('training', {}).get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw').lower()

        params = self.model.parameters()

        if opt_type == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=opt_config.get('learning_rate', 0.001),
                weight_decay=opt_config.get('weight_decay', 0.0001),
                betas=tuple(opt_config.get('betas', [0.9, 0.999]))
            )
        elif opt_type == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=opt_config.get('learning_rate', 0.001),
                weight_decay=opt_config.get('weight_decay', 0.0001),
                betas=tuple(opt_config.get('betas', [0.9, 0.999]))
            )
        elif opt_type == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=opt_config.get('learning_rate', 0.01),
                weight_decay=opt_config.get('weight_decay', 0.0001),
                momentum=opt_config.get('momentum', 0.9),
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        self.logger.info(f"Optimizer: {opt_type}, LR: {opt_config.get('learning_rate', 0.001)}")

    def _init_scheduler(self):
        """初始化学习率调度器"""
        sched_config = self.config.get('training', {}).get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine')

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=sched_type,
            epochs=self.epochs,
            warmup_epochs=sched_config.get('warmup_epochs', 5),
            min_lr=sched_config.get('min_lr', 1e-6),
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1),
            steps_per_epoch=len(self.train_loader)
        )

        self.logger.info(f"Scheduler: {sched_type}")

    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # 解包数据
            if len(batch) == 3:
                # 多源数据
                source1, source2, labels = batch
                source1 = source1.to(self.device)
                source2 = source2.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                with autocast(enabled=self.use_amp):
                    outputs, _ = self.model(source1, source2)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                # 单源数据
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
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

            # 统计
            total_loss += loss.item() * self.gradient_accumulation_steps * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in self.val_loader:
            if len(batch) == 3:
                source1, source2, labels = batch
                source1 = source1.to(self.device)
                source2 = source2.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.model(source1, source2)
            else:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total

        # 计算额外指标
        from sklearn.metrics import precision_score, recall_score, f1_score
        metrics = {
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }

        return avg_loss, accuracy, metrics

    def train(self) -> Dict:
        """完整训练流程"""
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, val_metrics = self.validate()

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)

            epoch_time = time.time() - epoch_start

            # 日志
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # 保存最佳模型
            is_best = val_loss < self.best_val_loss - self.early_stopping_min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0

                self._save_checkpoint('best_model.pth', is_best=True)
                self.logger.info(f"  New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # 定期保存
            if epoch % self.config.get('training', {}).get('checkpoint', {}).get('save_every', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # 早停检查
            if self.early_stopping_enabled:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        # 保存最终模型
        self._save_checkpoint('last_model.pth')

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.4f}")

        if self.writer is not None:
            self.writer.close()

        return self.history

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = os.path.join(self.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint.get('history', self.history)

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


# ============================================
# 消融实验
# ============================================

class AblationStudy:
    """消融实验管理器"""

    def __init__(
        self,
        base_config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: str
    ):
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir

        self.results = {}

    def run_experiment(
        self,
        experiment_name: str,
        model_config: Dict,
        epochs: int = 50
    ) -> Dict:
        """运行单个实验"""
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*50}")

        # 更新配置
        config = self.base_config.copy()
        config['model'].update(model_config)
        config['training']['epochs'] = epochs

        # 创建模型
        model = create_model(
            model_type=model_config.get('type', 'fusion_net'),
            traffic_dim=config['model']['source1_dim'],
            log_dim=config['model']['source2_dim'],
            num_classes=config['model']['num_classes'],
            config=model_config
        )

        # 设置日志
        exp_output_dir = os.path.join(self.output_dir, experiment_name)
        logger = setup_logger(exp_output_dir, experiment_name)

        # 训练
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=config,
            device=self.device,
            logger=logger,
            output_dir=exp_output_dir
        )

        history = trainer.train()

        # 测试
        test_loss, test_acc, test_metrics = trainer.validate()

        result = {
            'name': experiment_name,
            'config': model_config,
            'best_val_loss': trainer.best_val_loss,
            'best_val_acc': trainer.best_val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_metrics': test_metrics,
            'history': history
        }

        self.results[experiment_name] = result
        return result

    def compare_fusion_methods(self):
        """比较不同融合方法"""
        fusion_types = ['attention', 'multi_head', 'gated', 'cross', 'concat']

        for ft in fusion_types:
            self.run_experiment(
                f'fusion_{ft}',
                {'fusion_type': ft}
            )

    def compare_encoders(self):
        """比较不同编码器"""
        encoder_types = ['mlp', 'cnn', 'lstm', 'transformer']

        for et in encoder_types:
            self.run_experiment(
                f'encoder_{et}',
                {'encoder_type': et}
            )

    def get_summary(self) -> str:
        """获取实验摘要"""
        summary = "\n" + "=" * 60 + "\n"
        summary += "Ablation Study Results\n"
        summary += "=" * 60 + "\n"

        for name, result in self.results.items():
            summary += f"\n{name}:\n"
            summary += f"  Best Val Acc: {result['best_val_acc']:.4f}\n"
            summary += f"  Test Acc: {result['test_acc']:.4f}\n"
            summary += f"  Test F1: {result['test_metrics']['f1']:.4f}\n"

        return summary


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Train network attack detection model")
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验名称
    experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, experiment_name)

    # 设置日志
    logger = setup_logger(output_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")

    # 加载数据
    import pickle
    data_path = os.path.join(args.data_dir, 'multi_source_data.pkl')

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please run data preprocessing first: python main.py --mode preprocess")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 更新配置
    config['model']['source1_dim'] = data.get('source1_dim', data['s1_train'].shape[1])
    config['model']['source2_dim'] = data.get('source2_dim', data['s2_train'].shape[1])
    config['model']['num_classes'] = data.get('num_classes', len(np.unique(data['y_train'])))

    # 创建数据加载器
    data_dict = {
        'X1_train': data['s1_train'], 'X1_val': data['s1_val'], 'X1_test': data['s1_test'],
        'X2_train': data['s2_train'], 'X2_val': data['s2_val'], 'X2_test': data['s2_test'],
        'y_train': data['y_train'], 'y_val': data['y_val'], 'y_test': data['y_test']
    }

    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['loader'].get('num_workers', 4),
        use_weighted_sampler=config['data']['loader'].get('use_weighted_sampler', False),
        augment_train=config['data']['loader'].get('augment_train', False)
    )

    if args.ablation:
        # 消融实验
        ablation = AblationStudy(
            base_config=config,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            test_loader=loaders['test'],
            device=device,
            output_dir=os.path.join(output_dir, 'ablation')
        )

        ablation.compare_fusion_methods()
        ablation.compare_encoders()

        summary = ablation.get_summary()
        logger.info(summary)

        # 保存结果
        import json
        with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
            # 转换为可序列化格式
            serializable_results = {}
            for name, result in ablation.results.items():
                serializable_results[name] = {
                    'name': result['name'],
                    'best_val_acc': float(result['best_val_acc']),
                    'test_acc': float(result['test_acc']),
                    'test_metrics': {k: float(v) for k, v in result['test_metrics'].items()}
                }
            json.dump(serializable_results, f, indent=2)

    else:
        # 正常训练
        model_config = config.get('model', {})
        model = create_model(
            model_type=model_config.get('type', 'fusion_net'),
            traffic_dim=config['model']['source1_dim'],
            log_dim=config['model']['source2_dim'],
            num_classes=config['model']['num_classes'],
            config=model_config.get('architecture', {})
        )

        logger.info(f"Model: {model_config.get('type', 'fusion_net')}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        trainer = Trainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            config=config,
            device=device,
            logger=logger,
            output_dir=output_dir
        )

        if args.resume:
            trainer.load_checkpoint(args.resume)

        history = trainer.train()

        # 测试
        logger.info("\nEvaluating on test set...")
        test_loss, test_acc, test_metrics = trainer.validate()
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
