#!/usr/bin/env python3
"""
网络攻击检测系统 - 主入口

基于多源数据融合的网络攻击检测系统

使用方法:
    # 完整流程（预处理 + 训练 + 评估 + 可视化）
    python main.py --data_dir data/raw --mode full

    # 仅预处理数据
    python main.py --data_dir data/raw --mode preprocess

    # 仅训练模型
    python main.py --mode train

    # 仅评估模型
    python main.py --mode evaluate

    # 启动可视化仪表板
    python main.py --mode dashboard

    # 生成报告
    python main.py --mode report
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """加载配置文件"""
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_data(data_dir: str, config: dict):
    """数据预处理"""
    print("=" * 60)
    print("步骤 1: 数据预处理")
    print("=" * 60)

    from src.data.dataloader import (
        CICIDS2017Preprocessor, MultiSourceDataSplitter,
        DataBalancer, DataSplitter
    )

    # 1. 加载和预处理数据
    print(f"\n加载数据目录: {data_dir}")
    preprocessor = CICIDS2017Preprocessor()

    # 检查目录中的CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误: 在 {data_dir} 中未找到CSV文件")
        return None

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 加载数据
    if len(csv_files) == 1:
        data_path = os.path.join(data_dir, csv_files[0])
        features, labels = preprocessor.load_data(data_path)
    else:
        features, labels = preprocessor.load_multiple_files(data_dir)

    print(f"加载完成: {features.shape[0]} 样本, {features.shape[1]} 特征")

    # 2. 数据清洗
    print("\n清洗数据...")
    features, labels = preprocessor.clean_data(features, labels)
    print(f"清洗后: {features.shape[0]} 样本")

    # 3. 特征选择
    print("\n特征选择...")
    features, feature_names = preprocessor.select_features(
        features,
        method=config['data'].get('feature_selection_method', 'variance'),
        threshold=config['data'].get('variance_threshold', 0.01)
    )
    print(f"选择后: {features.shape[1]} 特征")

    # 4. 标签编码
    print("\n标签编码...")
    labels, class_names = preprocessor.encode_labels(
        labels,
        binary=config['data'].get('binary_classification', False)
    )
    print(f"类别: {class_names}")

    # 5. 特征归一化
    print("\n特征归一化...")
    features = preprocessor.normalize_features(features)

    # 6. 多源数据分割
    print("\n分割为多源数据...")
    splitter = MultiSourceDataSplitter()
    source1, source2 = splitter.split(features, feature_names)
    print(f"源1维度: {source1.shape[1]}, 源2维度: {source2.shape[1]}")

    # 7. 数据集分割
    print("\n划分训练/验证/测试集...")
    data_splitter = DataSplitter(
        train_ratio=config['data'].get('train_ratio', 0.7),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.2)
    )

    # 单源数据分割
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split(features, labels)

    # 多源数据分割
    (s1_train, s1_val, s1_test, s2_train, s2_val, s2_test,
     y_train_m, y_val_m, y_test_m) = data_splitter.split_multi_source(source1, source2, labels)

    # 8. 数据平衡（可选）
    if config['data'].get('balance_data', False):
        print("\n数据平衡...")
        balancer = DataBalancer(method=config['data'].get('balance_method', 'smote'))
        X_train, y_train = balancer.fit_resample(X_train, y_train)
        s1_train, y_train_m = balancer.fit_resample(s1_train, y_train_m)
        # s2_train不需要重新采样，使用相同索引
        print(f"平衡后训练集: {X_train.shape[0]} 样本")

    # 9. 保存处理后的数据
    output_dir = os.path.join(project_root, config['data'].get('processed_path', 'data/processed'))
    os.makedirs(output_dir, exist_ok=True)

    # 单源数据
    single_source_data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': feature_names,
        'class_names': class_names,
        'num_features': features.shape[1],
        'num_classes': len(class_names)
    }

    single_path = os.path.join(output_dir, 'single_source_data.pkl')
    with open(single_path, 'wb') as f:
        pickle.dump(single_source_data, f)
    print(f"\n单源数据已保存: {single_path}")

    # 多源数据
    multi_source_data = {
        's1_train': s1_train, 's1_val': s1_val, 's1_test': s1_test,
        's2_train': s2_train, 's2_val': s2_val, 's2_test': s2_test,
        'y_train': y_train_m, 'y_val': y_val_m, 'y_test': y_test_m,
        'source1_names': splitter.get_source1_names(feature_names),
        'source2_names': splitter.get_source2_names(feature_names),
        'class_names': class_names,
        'source1_dim': source1.shape[1],
        'source2_dim': source2.shape[1],
        'num_classes': len(class_names)
    }

    multi_path = os.path.join(output_dir, 'multi_source_data.pkl')
    with open(multi_path, 'wb') as f:
        pickle.dump(multi_source_data, f)
    print(f"多源数据已保存: {multi_path}")

    # 保存预处理器
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': preprocessor.scaler,
            'label_encoder': preprocessor.label_encoder,
            'feature_names': feature_names,
            'class_names': class_names
        }, f)
    print(f"预处理器已保存: {preprocessor_path}")

    print("\n数据预处理完成!")
    return single_source_data, multi_source_data


def train_model(config: dict):
    """训练模型"""
    print("=" * 60)
    print("步骤 2: 模型训练")
    print("=" * 60)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.models.fusion_net import FusionNet, SingleSourceNet
    from src.data.dataset import MultiSourceDataset, NetworkAttackDataset
    from src.visualization.monitor import TrainingMonitor

    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    processed_dir = os.path.join(project_root, config['data'].get('processed_path', 'data/processed'))

    multi_path = os.path.join(processed_dir, 'multi_source_data.pkl')
    with open(multi_path, 'rb') as f:
        data = pickle.load(f)

    # 创建数据集
    train_dataset = MultiSourceDataset(
        data['s1_train'], data['s2_train'], data['y_train']
    )
    val_dataset = MultiSourceDataset(
        data['s1_val'], data['s2_val'], data['y_val']
    )

    # 创建数据加载器
    batch_size = config['training'].get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    model_config = config['model']
    model = FusionNet(
        traffic_dim=data['source1_dim'],
        log_dim=data['source2_dim'],
        num_classes=data['num_classes'],
        hidden_dim=model_config.get('hidden_dim', 256),
        dropout=model_config.get('dropout', 0.3)
    ).to(device)

    print(f"\n模型结构:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training'].get('learning_rate', 0.001),
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 训练监控器
    experiment_name = f"fusion_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    monitor = TrainingMonitor(
        experiment_name=experiment_name,
        log_dir=os.path.join(project_root, 'outputs/logs'),
        checkpoint_dir=os.path.join(project_root, 'outputs/checkpoints'),
        early_stopping_patience=config['training'].get('early_stopping_patience', 15)
    )

    # 开始训练
    epochs = config['training'].get('epochs', 100)
    monitor.on_train_begin(config)

    print(f"\n开始训练 ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (s1, s2, labels) in enumerate(train_loader):
            s1, s2, labels = s1.to(device), s2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, attention = model(s1, s2)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for s1, s2, labels in val_loader:
                s1, s2, labels = s1.to(device), s2.to(device), labels.to(device)

                outputs, _ = model(s1, s2)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录训练结果
        should_stop = monitor.on_epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=current_lr,
            model=model,
            optimizer=optimizer
        )

        if should_stop:
            print("早停触发，停止训练")
            break

    # 训练结束
    summary = monitor.on_train_end()

    print("\n模型训练完成!")
    return model, monitor.experiment_name


def evaluate_model(config: dict, experiment_name: str = None):
    """评估模型"""
    print("=" * 60)
    print("步骤 3: 模型评估")
    print("=" * 60)

    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    from src.models.fusion_net import FusionNet
    from src.data.dataset import MultiSourceDataset
    from src.visualization.monitor import TrainingMonitor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    processed_dir = os.path.join(project_root, config['data'].get('processed_path', 'data/processed'))
    multi_path = os.path.join(processed_dir, 'multi_source_data.pkl')

    with open(multi_path, 'rb') as f:
        data = pickle.load(f)

    # 创建测试数据集
    test_dataset = MultiSourceDataset(
        data['s1_test'], data['s2_test'], data['y_test']
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载最佳模型
    if experiment_name is None:
        checkpoint_dir = os.path.join(project_root, 'outputs/checkpoints')
        experiments = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
        if not experiments:
            print("错误: 没有找到训练好的模型")
            return None
        experiment_name = sorted(experiments)[-1]

    model_path = os.path.join(project_root, 'outputs/checkpoints', experiment_name, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None

    # 加载模型
    model = FusionNet(
        traffic_dim=data['source1_dim'],
        log_dim=data['source2_dim'],
        num_classes=data['num_classes'],
        hidden_dim=config['model'].get('hidden_dim', 256),
        dropout=config['model'].get('dropout', 0.3)
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n加载模型: {model_path}")

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
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

    print("\n评估结果:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 保存结果
    results_dir = os.path.join(project_root, 'outputs/results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # 使用监控器保存结果
    monitor = TrainingMonitor(experiment_name=experiment_name)
    monitor.results_dir = results_dir
    monitor.save_model_results(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=data['class_names'],
        attention_weights=attention_weights,
        metrics=metrics
    )

    print("\n模型评估完成!")
    return metrics


def generate_report(config: dict, experiment_name: str = None):
    """生成实验报告"""
    print("=" * 60)
    print("步骤 4: 生成报告")
    print("=" * 60)

    from src.visualization.report import generate_full_report

    processed_dir = os.path.join(project_root, config['data'].get('processed_path', 'data/processed'))

    # 找到最新的实验
    if experiment_name is None:
        results_dir = os.path.join(project_root, 'outputs/results')
        if os.path.exists(results_dir):
            experiments = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            if experiments:
                experiment_name = sorted(experiments)[-1]

    if experiment_name is None:
        experiment_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    data_path = os.path.join(processed_dir, 'single_source_data.pkl')
    results_path = os.path.join(project_root, 'outputs/results', experiment_name, 'model_results.pkl')
    history_path = os.path.join(project_root, 'outputs/results', experiment_name, 'training_history.pkl')

    report_path = generate_full_report(
        experiment_name=experiment_name,
        data_path=data_path if os.path.exists(data_path) else None,
        results_path=results_path if os.path.exists(results_path) else None,
        history_path=history_path if os.path.exists(history_path) else None,
        output_dir=os.path.join(project_root, 'outputs/reports')
    )

    print(f"\n报告已生成: {report_path}")
    return report_path


def launch_dashboard():
    """启动可视化仪表板"""
    print("=" * 60)
    print("启动可视化仪表板")
    print("=" * 60)

    import subprocess

    app_path = os.path.join(project_root, 'src/visualization/app.py')
    print(f"\n启动 Streamlit 应用: {app_path}")
    print("在浏览器中访问: http://localhost:8501")

    subprocess.run(['streamlit', 'run', app_path])


def main():
    parser = argparse.ArgumentParser(
        description='网络攻击检测系统 - 基于多源数据融合的深度学习方法',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default='data/raw',
        help='原始数据目录路径 (包含CSV文件)'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'preprocess', 'train', 'evaluate', 'report', 'dashboard'],
        default='full',
        help='运行模式: full(完整流程), preprocess(仅预处理), train(仅训练), evaluate(仅评估), report(生成报告), dashboard(启动仪表板)'
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

    args = parser.parse_args()

    # 打印欢迎信息
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "网络攻击检测系统 v1.0" + " " * 27 + "║")
    print("║" + " " * 10 + "基于多源数据融合的深度学习方法" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    # 加载配置
    config = load_config(args.config)

    # 根据模式执行
    if args.mode == 'dashboard':
        launch_dashboard()
        return

    experiment_name = args.experiment

    if args.mode in ['full', 'preprocess']:
        # 数据预处理
        data_dir = os.path.join(project_root, args.data_dir)
        if not os.path.exists(data_dir):
            print(f"错误: 数据目录不存在: {data_dir}")
            print(f"请将CIC-IDS-2017数据集放入 {data_dir} 目录")
            return
        preprocess_data(data_dir, config)

    if args.mode in ['full', 'train']:
        # 训练模型
        model, experiment_name = train_model(config)

    if args.mode in ['full', 'evaluate']:
        # 评估模型
        evaluate_model(config, experiment_name)

    if args.mode in ['full', 'report']:
        # 生成报告
        generate_report(config, experiment_name)

    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)

    if args.mode == 'full':
        print("\n下一步操作:")
        print("  1. 查看报告: outputs/reports/<experiment>/report.html")
        print("  2. 启动仪表板: python main.py --mode dashboard")
        print("  3. 查看TensorBoard: tensorboard --logdir outputs/logs")


if __name__ == '__main__':
    main()
