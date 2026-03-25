"""
Windows 快速测试脚本

使用 KDD Cup 数据集进行快速测试
支持自动下载数据、预处理、训练和可视化

使用方法:
    # 完整测试流程
    python quick_test.py

    # 仅预处理
    python quick_test.py --mode preprocess

    # 仅训练
    python quick_test.py --mode train

    # 启动可视化
    python quick_test.py --mode dashboard

    # 指定数据文件
    python quick_test.py --data_file "path/to/kddcup.csv"
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import numpy as np


def setup_environment():
    """设置环境"""
    # 创建必要的目录
    dirs = [
        'data/raw',
        'data/processed',
        'outputs/checkpoints',
        'outputs/results',
        'outputs/logs',
        'outputs/figures'
    ]
    for d in dirs:
        os.makedirs(os.path.join(project_root, d), exist_ok=True)
    print("环境设置完成")


def download_or_find_data(data_file: str = None) -> str:
    """下载或查找数据文件"""
    from src.data.kddcup_loader import download_kddcup_sample

    if data_file and os.path.exists(data_file):
        print(f"使用指定数据文件: {data_file}")
        return data_file

    # 检查常见位置
    possible_paths = [
        os.path.join(project_root, 'data/raw/kddcup_10percent.csv'),
        os.path.join(project_root, 'data/raw/kddcup.data_10_percent'),
        os.path.join(project_root, 'data/raw/kddcup.csv'),
        os.path.join(project_root, 'data/raw/kddcup99.csv'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            return path

    # 尝试下载
    print("未找到本地数据文件，尝试下载...")
    downloaded_path = download_kddcup_sample()

    if downloaded_path and os.path.exists(downloaded_path):
        return downloaded_path

    print("\n" + "=" * 50)
    print("无法自动下载数据，请手动下载 KDD Cup 99 数据集")
    print("下载地址: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html")
    print("将数据文件放置到: data/raw/ 目录")
    print("=" * 50)
    return None


def preprocess_kddcup(data_file: str, sample_size: int = 50000):
    """预处理 KDD Cup 数据"""
    print("\n" + "=" * 60)
    print("步骤 1: 数据预处理 (KDD Cup 99)")
    print("=" * 60)

    from src.data.kddcup_loader import KDDCupPreprocessor, KDDMultiSourceSplitter
    from src.data.dataloader import DataSplitter

    # 1. 预处理
    preprocessor = KDDCupPreprocessor()
    result = preprocessor.preprocess(
        data_path=data_file,
        binary_classification=False,  # 5分类
        use_categories=True,
        normalize=True,
        sample_size=sample_size
    )

    X = result['X']
    y = result['y']
    feature_names = result['feature_names']
    class_names = result['class_names']

    print(f"\n预处理完成: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(class_names)} 类别")

    # 2. 多源数据分割
    print("\n分割为多源数据...")
    splitter = KDDMultiSourceSplitter(
        source1_groups=['basic', 'content'],
        source2_groups=['traffic', 'host']
    )
    X1, X2, names1, names2 = splitter.split(X, feature_names)

    # 3. 数据集划分
    print("\n划分训练/验证/测试集...")
    data_splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
    split_data = data_splitter.split_multi_source(X1, X2, y, stratify=True)

    # 4. 保存数据
    output_dir = os.path.join(project_root, 'data/processed')

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
    print(f"多源数据已保存: {multi_path}")

    # 单源数据
    single_split = data_splitter.split(X, y, stratify=True)
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
    print(f"单源数据已保存: {single_path}")

    print("\n数据预处理完成!")
    return multi_source_data


def train_model(epochs: int = 30, batch_size: int = 64):
    """训练模型"""
    print("\n" + "=" * 60)
    print("步骤 2: 模型训练")
    print("=" * 60)

    import torch
    from src.models.fusion_net import create_model
    from src.data.dataset import create_multi_source_loaders
    from src.train import Trainer, setup_logger

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    data_path = os.path.join(project_root, 'data/processed/multi_source_data.pkl')
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 {data_path}")
        print("请先运行预处理: python quick_test.py --mode preprocess")
        return None

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"加载数据: 源1维度={data['source1_dim']}, 源2维度={data['source2_dim']}, 类别数={data['num_classes']}")

    # 创建数据加载器
    data_dict = {
        'X1_train': data['s1_train'], 'X1_val': data['s1_val'], 'X1_test': data['s1_test'],
        'X2_train': data['s2_train'], 'X2_val': data['s2_val'], 'X2_test': data['s2_test'],
        'y_train': data['y_train'], 'y_val': data['y_val'], 'y_test': data['y_test']
    }

    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=batch_size,
        num_workers=0,  # Windows 兼容
        pin_memory=False
    )

    # 创建模型
    model = create_model(
        model_type='fusion_net',
        traffic_dim=data['source1_dim'],
        log_dim=data['source2_dim'],
        num_classes=data['num_classes'],
        config={
            'hidden_dim': 128,
            'dropout': 0.3,
            'encoder_type': 'mlp',
            'fusion_type': 'attention',
            'num_layers': 2,
            'num_heads': 4
        }
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 实验目录
    experiment_name = f"kddcup_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(project_root, 'outputs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 配置
    config = {
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': {'type': 'adamw', 'learning_rate': 0.001, 'weight_decay': 0.0001},
            'scheduler': {'type': 'cosine', 'warmup_epochs': 3, 'min_lr': 1e-6},
            'early_stopping': {'enabled': True, 'patience': 10, 'min_delta': 0.001},
            'loss': {'type': 'cross_entropy'},
            'gradient_clip': {'max_norm': 1.0},
            'mixed_precision': False,
            'checkpoint': {'save_every': 10}
        },
        'model': {'num_classes': data['num_classes']}
    }

    # 训练
    logger = setup_logger(output_dir, 'train')
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=config,
        device=device,
        logger=logger,
        output_dir=output_dir
    )

    history = trainer.train()

    print(f"\n训练完成!")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"最佳验证准确率: {trainer.best_val_acc:.4f}")
    print(f"输出目录: {output_dir}")

    # 测试集评估
    print("\n在测试集上评估...")
    test_loss, test_acc, test_metrics = trainer.evaluate(loaders['test'])
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试F1: {test_metrics['f1']:.4f}")

    # 保存结果
    results = {
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_metrics': test_metrics,
        'class_names': data['class_names']
    }

    results_path = os.path.join(output_dir, 'results', 'test_results.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    return experiment_name


def launch_dashboard():
    """启动可视化仪表板"""
    print("\n" + "=" * 60)
    print("启动可视化仪表板")
    print("=" * 60)

    import subprocess

    app_path = os.path.join(project_root, 'src/visualization/app.py')
    print(f"Streamlit 应用: {app_path}")
    print("访问地址: http://localhost:8501")
    print("\n按 Ctrl+C 停止服务")

    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])


def main():
    parser = argparse.ArgumentParser(
        description='KDD Cup 数据集快速测试 (Windows)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'preprocess', 'train', 'dashboard'],
        default='full',
        help='运行模式 (默认: full)'
    )

    parser.add_argument(
        '--data_file', '-d',
        type=str,
        default=None,
        help='KDD Cup 数据文件路径'
    )

    parser.add_argument(
        '--sample_size', '-s',
        type=int,
        default=50000,
        help='采样数量 (默认: 50000)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=30,
        help='训练轮数 (默认: 30)'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=64,
        help='批次大小 (默认: 64)'
    )

    args = parser.parse_args()

    # 打印欢迎信息
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "KDD Cup 快速测试工具" + " " * 26 + "║")
    print("║" + " " * 12 + "Windows 适配版本" + " " * 30 + "║")
    print("╚" + "═" * 58 + "╝\n")

    # 设置环境
    setup_environment()

    if args.mode == 'dashboard':
        launch_dashboard()
        return

    # 查找或下载数据
    if args.mode in ['full', 'preprocess']:
        data_file = download_or_find_data(args.data_file)
        if not data_file:
            return

        preprocess_kddcup(data_file, sample_size=args.sample_size)

    if args.mode in ['full', 'train']:
        train_model(epochs=args.epochs, batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    if args.mode == 'full':
        print("\n下一步操作:")
        print("  1. 启动可视化: python quick_test.py --mode dashboard")
        print("  2. 查看结果: outputs/ 目录")


if __name__ == '__main__':
    main()
