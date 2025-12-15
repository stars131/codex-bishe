#!/usr/bin/env python3
"""
网络攻击检测系统 - 主入口

基于多源数据融合的网络攻击检测系统

使用方法:
    # 完整流程（预处理 + 训练 + 评估 + 报告）
    python main.py --data_dir "/path/to/CIC-IDS-2017" --mode full

    # 仅预处理数据
    python main.py --data_dir "/path/to/CIC-IDS-2017" --mode preprocess

    # 仅训练模型
    python main.py --mode train

    # 仅评估模型
    python main.py --mode evaluate

    # 启动可视化仪表板
    python main.py --mode dashboard

    # 生成报告
    python main.py --mode report

    # 消融实验
    python main.py --mode ablation

Author: Network Attack Detection Project
"""

import os
import sys
import argparse
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# 添加项目路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def setup_logging(log_dir: str = None) -> logging.Logger:
    """设置日志"""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    else:
        log_file = None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """加载配置文件"""
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_data(data_dir: str, config: dict, logger: logging.Logger):
    """
    数据预处理

    Args:
        data_dir: 原始数据目录
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("步骤 1: 数据预处理")
    logger.info("=" * 60)

    from src.data.dataloader import (
        CICIDS2017Preprocessor, MultiSourceDataSplitter, DataSplitter
    )

    # 1. 加载和预处理数据
    logger.info(f"数据目录: {data_dir}")

    preprocessor = CICIDS2017Preprocessor(config)

    # 预处理配置
    preprocess_config = config.get('data', {}).get('preprocessing', {})

    # 执行预处理
    result = preprocessor.preprocess(
        data_path=data_dir,
        binary_classification=preprocess_config.get('binary_classification', False),
        feature_selection=preprocess_config.get('feature_selection', 'correlation'),
        normalize=preprocess_config.get('normalize', True)
    )

    X = result['X']
    y = result['y']
    feature_names = result['feature_names']
    class_names = result['class_names']

    logger.info(f"预处理完成: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(class_names)} 类别")

    # 2. 多源数据分割
    logger.info("分割为多源数据...")
    ms_config = config.get('data', {}).get('multi_source', {})
    splitter = MultiSourceDataSplitter(
        source1_groups=ms_config.get('source1_groups', ['traffic', 'temporal']),
        source2_groups=ms_config.get('source2_groups', ['flags', 'header', 'bulk'])
    )

    X1, X2, names1, names2 = splitter.split(X, feature_names)
    logger.info(f"源1: {X1.shape[1]} 特征, 源2: {X2.shape[1]} 特征")

    # 3. 数据集划分
    logger.info("划分训练/验证/测试集...")
    split_config = config.get('data', {}).get('split', {})
    data_splitter = DataSplitter(
        test_size=split_config.get('test_size', 0.2),
        val_size=split_config.get('val_size', 0.1),
        random_state=split_config.get('random_state', 42)
    )

    split_data = data_splitter.split_multi_source(X1, X2, y, stratify=split_config.get('stratify', True))

    logger.info(f"训练集: {len(split_data['y_train'])} 样本")
    logger.info(f"验证集: {len(split_data['y_val'])} 样本")
    logger.info(f"测试集: {len(split_data['y_test'])} 样本")

    # 4. 保存处理后的数据
    output_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    os.makedirs(output_dir, exist_ok=True)

    # 多源数据
    multi_source_data = {
        's1_train': split_data['X1_train'],
        's1_val': split_data['X1_val'],
        's1_test': split_data['X1_test'],
        's2_train': split_data['X2_train'],
        's2_val': split_data['X2_val'],
        's2_test': split_data['X2_test'],
        'y_train': split_data['y_train'],
        'y_val': split_data['y_val'],
        'y_test': split_data['y_test'],
        'source1_names': names1,
        'source2_names': names2,
        'class_names': class_names,
        'source1_dim': X1.shape[1],
        'source2_dim': X2.shape[1],
        'num_classes': len(class_names)
    }

    multi_path = os.path.join(output_dir, 'multi_source_data.pkl')
    with open(multi_path, 'wb') as f:
        pickle.dump(multi_source_data, f)
    logger.info(f"多源数据已保存: {multi_path}")

    # 单源数据（用于基线对比）
    single_split = data_splitter.split(X, y, stratify=split_config.get('stratify', True))
    single_source_data = {
        'X_train': single_split['X_train'],
        'X_val': single_split['X_val'],
        'X_test': single_split['X_test'],
        'y_train': single_split['y_train'],
        'y_val': single_split['y_val'],
        'y_test': single_split['y_test'],
        'feature_names': feature_names,
        'class_names': class_names,
        'num_features': X.shape[1],
        'num_classes': len(class_names)
    }

    single_path = os.path.join(output_dir, 'single_source_data.pkl')
    with open(single_path, 'wb') as f:
        pickle.dump(single_source_data, f)
    logger.info(f"单源数据已保存: {single_path}")

    # 保存预处理器
    preprocessor_data = {
        'scaler': preprocessor.scaler,
        'label_encoder': preprocessor.label_encoder,
        'feature_names': feature_names,
        'class_names': class_names
    }
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor_data, f)
    logger.info(f"预处理器已保存: {preprocessor_path}")

    logger.info("数据预处理完成!")
    return multi_source_data, single_source_data


def train_model(config: dict, logger: logging.Logger):
    """
    训练模型

    Args:
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("步骤 2: 模型训练")
    logger.info("=" * 60)

    import torch
    from src.models.fusion_net import create_model
    from src.data.dataset import create_multi_source_loaders
    from src.train import Trainer, setup_logger as train_setup_logger

    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载预处理数据
    processed_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    data_path = os.path.join(processed_dir, 'multi_source_data.pkl')

    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.error("请先运行预处理: python main.py --mode preprocess")
        return None, None

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"加载数据: {data_path}")
    logger.info(f"源1维度: {data['source1_dim']}, 源2维度: {data['source2_dim']}")
    logger.info(f"类别数: {data['num_classes']}")

    # 更新配置
    config['model']['source1_dim'] = data['source1_dim']
    config['model']['source2_dim'] = data['source2_dim']
    config['model']['num_classes'] = data['num_classes']

    # 创建数据加载器
    data_dict = {
        'X1_train': data['s1_train'], 'X1_val': data['s1_val'], 'X1_test': data['s1_test'],
        'X2_train': data['s2_train'], 'X2_val': data['s2_val'], 'X2_test': data['s2_test'],
        'y_train': data['y_train'], 'y_val': data['y_val'], 'y_test': data['y_test']
    }

    loader_config = config.get('data', {}).get('loader', {})
    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=loader_config.get('batch_size', 64),
        num_workers=loader_config.get('num_workers', 4),
        use_weighted_sampler=loader_config.get('use_weighted_sampler', False),
        augment_train=loader_config.get('augment_train', False)
    )

    # 创建模型
    model_config = config.get('model', {})
    arch_config = model_config.get('architecture', {})

    model = create_model(
        model_type=model_config.get('type', 'fusion_net'),
        traffic_dim=data['source1_dim'],
        log_dim=data['source2_dim'],
        num_classes=data['num_classes'],
        config={
            'hidden_dim': arch_config.get('hidden_dim', 256),
            'dropout': arch_config.get('dropout', 0.3),
            'encoder_type': arch_config.get('encoder_type', 'mlp'),
            'fusion_type': config.get('model', {}).get('fusion', {}).get('method', 'attention'),
            'num_layers': arch_config.get('num_layers', 2),
            'num_heads': config.get('model', {}).get('fusion', {}).get('attention_heads', 4)
        }
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params:,}")

    # 实验目录
    experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(project_root, 'outputs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 训练日志
    train_logger = train_setup_logger(output_dir, 'train')

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=config,
        device=device,
        logger=train_logger,
        output_dir=output_dir
    )

    # 开始训练
    history = trainer.train()

    logger.info("模型训练完成!")
    logger.info(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    logger.info(f"最佳验证准确率: {trainer.best_val_acc:.4f}")
    logger.info(f"输出目录: {output_dir}")

    return trainer, experiment_name


def evaluate_model(config: dict, experiment_name: str, logger: logging.Logger):
    """
    评估模型

    Args:
        config: 配置字典
        experiment_name: 实验名称
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("步骤 3: 模型评估")
    logger.info("=" * 60)

    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report

    from src.models.fusion_net import create_model
    from src.data.dataset import MultiSourceDataset
    from src.utils.helpers import evaluate_model as compute_metrics, print_metrics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    processed_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    data_path = os.path.join(processed_dir, 'multi_source_data.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 创建测试数据集
    test_dataset = MultiSourceDataset(
        data['s1_test'], data['s2_test'], data['y_test']
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 查找模型文件
    if experiment_name is None:
        checkpoint_dir = os.path.join(project_root, 'outputs')
        if os.path.exists(checkpoint_dir):
            experiments = [d for d in os.listdir(checkpoint_dir)
                           if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('exp_')]
            if experiments:
                experiment_name = sorted(experiments)[-1]

    if experiment_name is None:
        logger.error("未找到训练好的模型")
        return None

    model_path = os.path.join(project_root, 'outputs', experiment_name, 'checkpoints', 'best_model.pth')

    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None

    logger.info(f"加载模型: {model_path}")

    # 创建模型
    model_config = config.get('model', {})
    arch_config = model_config.get('architecture', {})

    model = create_model(
        model_type=model_config.get('type', 'fusion_net'),
        traffic_dim=data['source1_dim'],
        log_dim=data['source2_dim'],
        num_classes=data['num_classes'],
        config={
            'hidden_dim': arch_config.get('hidden_dim', 256),
            'dropout': arch_config.get('dropout', 0.3),
            'encoder_type': arch_config.get('encoder_type', 'mlp'),
            'fusion_type': config.get('model', {}).get('fusion', {}).get('method', 'attention'),
            'num_layers': arch_config.get('num_layers', 2),
            'num_heads': config.get('model', {}).get('fusion', {}).get('attention_heads', 4)
        }
    )

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 评估
    all_preds = []
    all_labels = []
    all_probs = []
    all_attention = []

    with torch.no_grad():
        for s1, s2, labels in test_loader:
            s1, s2, labels = s1.to(device), s2.to(device), labels.to(device)

            outputs, attention = model(s1, s2)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_attention.extend(attention.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    attention_weights = np.array(all_attention)

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names=data['class_names'])
    print_metrics(metrics, "测试集评估结果")

    # 分类报告
    logger.info("\n分类报告:")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=data['class_names'], zero_division=0))

    # 保存结果
    results_dir = os.path.join(project_root, 'outputs', experiment_name, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'attention_weights': attention_weights,
        'metrics': metrics,
        'class_names': data['class_names']
    }

    results_path = os.path.join(results_dir, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"评估结果已保存: {results_path}")

    logger.info("模型评估完成!")
    return metrics


def run_ablation(config: dict, logger: logging.Logger):
    """
    运行消融实验

    Args:
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("消融实验")
    logger.info("=" * 60)

    import torch
    from src.data.dataset import create_multi_source_loaders
    from src.train import AblationStudy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    processed_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    data_path = os.path.join(processed_dir, 'multi_source_data.pkl')

    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.error("请先运行预处理: python main.py --mode preprocess")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 更新配置
    config['model']['source1_dim'] = data['source1_dim']
    config['model']['source2_dim'] = data['source2_dim']
    config['model']['num_classes'] = data['num_classes']

    # 创建数据加载器
    data_dict = {
        'X1_train': data['s1_train'], 'X1_val': data['s1_val'], 'X1_test': data['s1_test'],
        'X2_train': data['s2_train'], 'X2_val': data['s2_val'], 'X2_test': data['s2_test'],
        'y_train': data['y_train'], 'y_val': data['y_val'], 'y_test': data['y_test']
    }

    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=config.get('data', {}).get('loader', {}).get('batch_size', 64),
        num_workers=0
    )

    # 实验目录
    experiment_name = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(project_root, 'outputs', experiment_name)

    # 创建消融实验
    ablation = AblationStudy(
        base_config=config,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        test_loader=loaders['test'],
        device=device,
        output_dir=output_dir
    )

    # 比较融合方法
    ablation_config = config.get('ablation', {})

    if ablation_config.get('compare_fusion', {}).get('enabled', True):
        logger.info("比较融合方法...")
        ablation.compare_fusion_methods()

    if ablation_config.get('compare_sources', {}).get('enabled', True):
        logger.info("比较编码器...")
        ablation.compare_encoders()

    # 输出摘要
    summary = ablation.get_summary()
    logger.info(summary)

    # 保存结果
    import json
    results_path = os.path.join(output_dir, 'ablation_results.json')
    os.makedirs(output_dir, exist_ok=True)

    serializable_results = {}
    for name, result in ablation.results.items():
        serializable_results[name] = {
            'best_val_acc': float(result['best_val_acc']),
            'test_acc': float(result['test_acc']),
            'test_f1': float(result['test_metrics']['f1'])
        }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"消融实验结果已保存: {results_path}")


def generate_report(config: dict, experiment_name: str, logger: logging.Logger):
    """
    生成实验报告

    Args:
        config: 配置字典
        experiment_name: 实验名称
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("步骤 4: 生成报告")
    logger.info("=" * 60)

    from src.visualization.report import generate_full_report

    # 查找实验
    if experiment_name is None:
        results_dir = os.path.join(project_root, 'outputs')
        if os.path.exists(results_dir):
            experiments = [d for d in os.listdir(results_dir)
                           if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('exp_')]
            if experiments:
                experiment_name = sorted(experiments)[-1]

    if experiment_name is None:
        experiment_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 路径
    processed_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    data_path = os.path.join(processed_dir, 'single_source_data.pkl')
    results_path = os.path.join(project_root, 'outputs', experiment_name, 'results', 'test_results.pkl')
    history_path = os.path.join(project_root, 'outputs', experiment_name, 'checkpoints', 'best_model.pth')

    report_path = generate_full_report(
        experiment_name=experiment_name,
        data_path=data_path if os.path.exists(data_path) else None,
        results_path=results_path if os.path.exists(results_path) else None,
        history_path=history_path if os.path.exists(history_path) else None,
        output_dir=os.path.join(project_root, 'outputs', experiment_name, 'reports')
    )

    logger.info(f"报告已生成: {report_path}")
    return report_path


def launch_dashboard(logger: logging.Logger):
    """启动可视化仪表板"""
    logger.info("=" * 60)
    logger.info("启动可视化仪表板")
    logger.info("=" * 60)

    import subprocess

    app_path = os.path.join(project_root, 'src/visualization/app.py')
    logger.info(f"Streamlit 应用: {app_path}")
    logger.info("访问地址: http://localhost:8501")

    subprocess.run(['streamlit', 'run', app_path])


def main():
    parser = argparse.ArgumentParser(
        description='网络攻击检测系统 - 基于多源数据融合的深度学习方法',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default=None,
        help='原始数据目录路径 (包含CIC-IDS-2017 CSV文件)'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'preprocess', 'train', 'evaluate', 'report', 'dashboard', 'ablation'],
        default='full',
        help='运行模式'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='src/config/config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default=None,
        help='实验名称 (用于evaluate和report模式)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 打印欢迎信息
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "网络攻击检测系统 v2.0" + " " * 27 + "║")
    print("║" + " " * 10 + "基于多源数据融合的深度学习方法" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝\n")

    # 设置日志
    log_dir = os.path.join(project_root, 'outputs', 'logs')
    logger = setup_logging(log_dir)

    # 设置随机种子
    from src.utils.helpers import set_seed
    set_seed(args.seed)
    logger.info(f"随机种子: {args.seed}")

    # 加载配置
    config = load_config(args.config)
    logger.info(f"配置文件: {args.config}")

    # 根据模式执行
    if args.mode == 'dashboard':
        launch_dashboard(logger)
        return

    experiment_name = args.experiment

    try:
        if args.mode in ['full', 'preprocess']:
            # 数据预处理
            if args.data_dir is None:
                data_dir = config.get('data', {}).get('raw_dir')
            else:
                data_dir = args.data_dir

            if data_dir is None or not os.path.exists(data_dir):
                logger.error(f"数据目录不存在: {data_dir}")
                logger.error("请使用 --data_dir 参数指定CIC-IDS-2017数据集路径")
                return

            preprocess_data(data_dir, config, logger)

        if args.mode in ['full', 'train']:
            # 训练模型
            trainer, experiment_name = train_model(config, logger)

        if args.mode in ['full', 'evaluate']:
            # 评估模型
            evaluate_model(config, experiment_name, logger)

        if args.mode in ['full', 'report']:
            # 生成报告
            generate_report(config, experiment_name, logger)

        if args.mode == 'ablation':
            # 消融实验
            run_ablation(config, logger)

        logger.info("\n" + "=" * 60)
        logger.info("所有任务完成!")
        logger.info("=" * 60)

        if args.mode == 'full':
            logger.info("\n下一步操作:")
            logger.info(f"  1. 查看报告: outputs/{experiment_name}/reports/")
            logger.info("  2. 启动仪表板: python main.py --mode dashboard")
            logger.info("  3. 查看TensorBoard: tensorboard --logdir outputs/")

    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
