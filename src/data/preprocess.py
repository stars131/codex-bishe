#!/usr/bin/env python3
"""
数据预处理主脚本

用于处理 CIC-IDS-2017 数据集，完成以下步骤：
1. 加载原始CSV数据
2. 数据清洗和预处理
3. 特征选择和标准化
4. 多源数据分割
5. 训练/验证/测试集划分
6. 数据分析报告生成
7. 保存处理后的数据

使用方法:
    python src/data/preprocess.py --data_dir <数据目录> [--binary] [--balance] [--analyze]
"""
import os
import sys
import argparse
import pickle
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.dataloader import (
    CICIDS2017Preprocessor,
    MultiSourceDataSplitter,
    DataBalancer,
    DataSplitter
)
from src.data.visualization import generate_data_report, DataAnalyzer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIC-IDS-2017 数据预处理脚本')

    # 数据路径
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/hgfs/linux-desktop共享文件夹/数据/CIC-IDS-2017',
        help='原始数据目录或CSV文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='输出目录'
    )

    # 预处理选项
    parser.add_argument(
        '--binary',
        action='store_true',
        help='使用二分类（正常 vs 攻击）'
    )
    parser.add_argument(
        '--feature_selection',
        type=str,
        default='correlation',
        choices=['all', 'variance', 'correlation'],
        help='特征选择方法'
    )

    # 数据平衡
    parser.add_argument(
        '--balance',
        action='store_true',
        help='是否对训练数据进行平衡采样'
    )
    parser.add_argument(
        '--balance_method',
        type=str,
        default='smote',
        choices=['smote', 'adasyn', 'undersample', 'smote_tomek'],
        help='数据平衡方法'
    )

    # 数据分割
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='测试集比例'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='验证集比例'
    )

    # 多源分割配置
    parser.add_argument(
        '--source1_groups',
        type=str,
        nargs='+',
        default=['traffic', 'temporal'],
        help='第一数据源包含的特征组'
    )
    parser.add_argument(
        '--source2_groups',
        type=str,
        nargs='+',
        default=['flags', 'header', 'bulk'],
        help='第二数据源包含的特征组'
    )

    # 可视化
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='生成数据分析报告'
    )
    parser.add_argument(
        '--figures_dir',
        type=str,
        default='outputs/figures',
        help='图表保存目录'
    )

    # 其他
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    print("=" * 60)
    print("CIC-IDS-2017 数据预处理")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  分类类型: {'二分类' if args.binary else '多分类'}")
    print(f"  特征选择: {args.feature_selection}")
    print(f"  数据平衡: {args.balance} ({args.balance_method if args.balance else 'N/A'})")
    print(f"  测试集比例: {args.test_size}")
    print(f"  验证集比例: {args.val_size}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ========== 步骤1: 加载和预处理数据 ==========
    print("\n" + "=" * 40)
    print("步骤 1: 加载和预处理数据")
    print("=" * 40)

    preprocessor = CICIDS2017Preprocessor()

    # 检查数据路径
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据路径不存在 - {args.data_dir}")
        print("\n请确保数据文件已放置在正确位置。")
        print("支持的数据格式:")
        print("  - 单个CSV文件")
        print("  - 包含多个CSV文件的目录")
        sys.exit(1)

    # 预处理
    result = preprocessor.preprocess(
        data_path=args.data_dir,
        binary_classification=args.binary,
        feature_selection=args.feature_selection,
        normalize=True
    )

    X = result['X']
    y = result['y']
    feature_names = result['feature_names']
    class_names = result['class_names']

    # ========== 步骤2: 多源数据分割 ==========
    print("\n" + "=" * 40)
    print("步骤 2: 多源数据分割")
    print("=" * 40)

    splitter = MultiSourceDataSplitter(
        source1_groups=args.source1_groups,
        source2_groups=args.source2_groups
    )

    X1, X2, names1, names2 = splitter.split(X, feature_names)

    # ========== 步骤3: 数据集划分 ==========
    print("\n" + "=" * 40)
    print("步骤 3: 数据集划分")
    print("=" * 40)

    data_splitter = DataSplitter(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )

    # 单源数据划分
    single_source_data = data_splitter.split(X, y)

    # 多源数据划分
    multi_source_data = data_splitter.split_multi_source(X1, X2, y)

    # ========== 步骤4: 数据平衡（可选） ==========
    if args.balance:
        print("\n" + "=" * 40)
        print("步骤 4: 数据平衡")
        print("=" * 40)

        balancer = DataBalancer(method=args.balance_method, random_state=args.seed)

        # 平衡单源训练数据
        single_source_data['X_train'], single_source_data['y_train'] = balancer.balance(
            single_source_data['X_train'],
            single_source_data['y_train']
        )

        # 平衡多源训练数据（使用相同的索引）
        # 需要将两个源的数据拼接起来一起平衡
        X_train_combined = np.hstack([
            multi_source_data['X1_train'],
            multi_source_data['X2_train']
        ])
        X_balanced, y_balanced = balancer.balance(
            X_train_combined,
            multi_source_data['y_train']
        )

        # 分开
        dim1 = multi_source_data['X1_train'].shape[1]
        multi_source_data['X1_train'] = X_balanced[:, :dim1]
        multi_source_data['X2_train'] = X_balanced[:, dim1:]
        multi_source_data['y_train'] = y_balanced

    # ========== 步骤5: 保存数据 ==========
    print("\n" + "=" * 40)
    print("步骤 5: 保存预处理数据")
    print("=" * 40)

    # 保存单源数据
    single_source_path = os.path.join(args.output_dir, 'single_source_data.pkl')
    single_source_save = {
        **single_source_data,
        'feature_names': feature_names,
        'class_names': class_names,
        'num_features': X.shape[1],
        'num_classes': len(class_names)
    }
    with open(single_source_path, 'wb') as f:
        pickle.dump(single_source_save, f)
    print(f"单源数据已保存: {single_source_path}")

    # 保存多源数据
    multi_source_path = os.path.join(args.output_dir, 'multi_source_data.pkl')
    multi_source_save = {
        **multi_source_data,
        'source1_names': names1,
        'source2_names': names2,
        'class_names': class_names,
        'source1_dim': X1.shape[1],
        'source2_dim': X2.shape[1],
        'num_classes': len(class_names)
    }
    with open(multi_source_path, 'wb') as f:
        pickle.dump(multi_source_save, f)
    print(f"多源数据已保存: {multi_source_path}")

    # 保存预处理器（用于推理时的数据处理）
    preprocessor_path = os.path.join(args.output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': preprocessor.scaler,
            'label_encoder': preprocessor.label_encoder,
            'feature_names': feature_names,
            'class_names': class_names
        }, f)
    print(f"预处理器已保存: {preprocessor_path}")

    # 保存配置
    config_path = os.path.join(args.output_dir, 'data_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump({
            'binary': args.binary,
            'feature_selection': args.feature_selection,
            'balance': args.balance,
            'balance_method': args.balance_method if args.balance else None,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'source1_groups': args.source1_groups,
            'source2_groups': args.source2_groups,
            'seed': args.seed
        }, f)
    print(f"配置已保存: {config_path}")

    # ========== 步骤6: 数据分析报告（可选） ==========
    if args.analyze:
        print("\n" + "=" * 40)
        print("步骤 6: 生成数据分析报告")
        print("=" * 40)

        generate_data_report(
            X, y, feature_names, class_names,
            save_dir=args.figures_dir
        )

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("数据预处理完成!")
    print("=" * 60)

    print(f"\n输出文件:")
    print(f"  - {single_source_path}")
    print(f"  - {multi_source_path}")
    print(f"  - {preprocessor_path}")
    print(f"  - {config_path}")

    print(f"\n数据统计:")
    print(f"  总样本数: {len(y):,}")
    print(f"  训练集: {len(single_source_data['y_train']):,}")
    print(f"  验证集: {len(single_source_data['y_val']):,}")
    print(f"  测试集: {len(single_source_data['y_test']):,}")
    print(f"  单源特征维度: {X.shape[1]}")
    print(f"  多源-Source1 维度: {X1.shape[1]}")
    print(f"  多源-Source2 维度: {X2.shape[1]}")
    print(f"  类别数: {len(class_names)}")

    print("\n下一步:")
    print("  运行训练脚本: python src/train.py --data_path data/processed/multi_source_data.pkl")


if __name__ == '__main__':
    main()
