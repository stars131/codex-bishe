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
import re
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from copy import deepcopy

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


def _collect_source_arrays(
    data: dict,
    split: str
) -> list:
    source_indices = sorted({
        int(match.group(1))
        for key in data.keys()
        for match in [re.match(rf'^s(\d+)_{split}$', key)]
        if match is not None
    })
    return [data[f's{idx}_{split}'] for idx in source_indices], source_indices


def _build_loader_data_dict(data: dict) -> tuple:
    _, source_indices = _collect_source_arrays(data, 'train')
    if len(source_indices) < 2:
        raise ValueError("multi_source_data.pkl 至少需要两个数据源（s1_*, s2_*）")

    data_dict = {
        'y_train': data['y_train'],
        'y_val': data['y_val'],
        'y_test': data['y_test']
    }
    for idx in source_indices:
        data_dict[f'X{idx}_train'] = data[f's{idx}_train']
        data_dict[f'X{idx}_val'] = data[f's{idx}_val']
        data_dict[f'X{idx}_test'] = data[f's{idx}_test']
    return data_dict, source_indices


def _infer_source_dims(data: dict, source_indices: list) -> list:
    return [data[f's{idx}_train'].shape[1] for idx in source_indices]


def _infer_model_source_dims(data: dict, source_indices: list) -> tuple:
    source_dims = _infer_source_dims(data, source_indices)
    return source_dims[0], source_dims[1]


def _build_model_factory_config(config: dict) -> tuple:
    model_config = config.get('model', {})
    arch_config = model_config.get('architecture', {})
    factory_config = {
        'hidden_dim': arch_config.get('hidden_dim', 256),
        'dropout': arch_config.get('dropout', 0.3),
        'encoder_type': arch_config.get('encoder_type', 'mlp'),
        'fusion_type': model_config.get('fusion', {}).get('method', 'attention'),
        'num_layers': arch_config.get('num_layers', 2),
        'num_heads': model_config.get('fusion', {}).get('attention_heads', 4),
        'source_dims': model_config.get('source_dims'),
        'decision_fusion': deepcopy(model_config.get('decision_fusion', {})),
        'agentic_mode': deepcopy(model_config.get('agentic_mode', {})),
    }
    return model_config, factory_config


def _validate_multi_source_groups(ms_config: dict, splitter, logger: logging.Logger):
    requested_groups = []
    requested_groups.extend(ms_config.get('source1_groups', []))
    requested_groups.extend(ms_config.get('source2_groups', []))
    for groups in ms_config.get('extra_source_groups', []):
        if isinstance(groups, str):
            requested_groups.append(groups)
        else:
            requested_groups.extend(groups)

    unknown_groups = sorted({
        group for group in requested_groups
        if group not in splitter.FEATURE_GROUPS
    })
    if unknown_groups:
        raise ValueError(
            "Unsupported feature groups in config.data.multi_source: "
            + ", ".join(unknown_groups)
            + ". Current CIC preprocessing only supports groups defined in "
            "MultiSourceDataSplitter.FEATURE_GROUPS."
        )

    if requested_groups:
        logger.info("Using feature groups: " + ", ".join(requested_groups))


def _validate_tabular_source_schema(data_frames: list, source_specs: list):
    if len(data_frames) <= 1:
        return

    normalized_frames = [
        [str(col).strip() for col in df.columns]
        for df in data_frames
    ]
    ref_name = source_specs[0].get('name', 'source_1')
    ref_cols = normalized_frames[0]
    ref_col_set = set(ref_cols)

    for idx in range(1, len(normalized_frames)):
        current_name = source_specs[idx].get('name', f'source_{idx + 1}')
        current_cols = normalized_frames[idx]
        if current_cols == ref_cols:
            continue

        current_col_set = set(current_cols)
        missing = sorted(ref_col_set - current_col_set)[:10]
        extra = sorted(current_col_set - ref_col_set)[:10]
        raise ValueError(
            f"Schema mismatch between '{ref_name}' and '{current_name}'. "
            "The current multi-source merge only supports CSV files with the same column schema. "
            f"Missing columns: {missing or 'none'}. Extra columns: {extra or 'none'}."
        )


def _merge_config_dicts(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_config_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_institutional_sources(
    primary_data_dir: str,
    config: dict,
    logger: logging.Logger
) -> list:
    seen_paths = set()
    sources = []

    if primary_data_dir:
        abs_path = os.path.abspath(primary_data_dir)
        if abs_path not in seen_paths:
            sources.append({'name': 'primary', 'path': primary_data_dir})
            seen_paths.add(abs_path)

    for item in config.get('data', {}).get('institutional_sources', []):
        if not isinstance(item, dict) or not item.get('enabled', False):
            continue
        path = item.get('path')
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen_paths:
            continue
        sources.append({
            'name': item.get('name', os.path.basename(path)),
            'institution': item.get('institution', ''),
            'path': path
        })
        seen_paths.add(abs_path)

    multimodal_config = config.get('data', {}).get('multimodal', {})
    if multimodal_config.get('enabled', False):
        input_format = multimodal_config.get('input_format', 'single_table')
        if input_format == 'single_table':
            mm_path = multimodal_config.get('path')
            if mm_path:
                abs_path = os.path.abspath(mm_path)
                if abs_path not in seen_paths:
                    sources.append({
                        'name': 'multimodal_processed',
                        'path': mm_path
                    })
                    seen_paths.add(abs_path)
        elif input_format == 'pre_split':
            split_specs = multimodal_config.get('splits', {})
            for split_name, split_spec in split_specs.items():
                if not isinstance(split_spec, dict):
                    continue
                for key in ('path', 'flow_path', 'log_path', 'label_path', 'id_path'):
                    split_path = split_spec.get(key)
                    if not split_path:
                        continue
                    abs_path = os.path.abspath(split_path)
                    if abs_path in seen_paths:
                        continue
                    sources.append({
                        'name': f'multimodal_{split_name}_{key}',
                        'path': split_path
                    })
                    seen_paths.add(abs_path)

    valid_sources = []
    for source in sources:
        if os.path.exists(source['path']):
            valid_sources.append(source)
        else:
            logger.warning(f"数据源路径不存在，已跳过: {source['path']}")

    return valid_sources


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

    multimodal_config = config.get('data', {}).get('multimodal', {})
    if multimodal_config.get('enabled', False):
        from src.data.multimodal_builder import MultimodalProcessedDataBuilder

        if data_dir and not multimodal_config.get('path') and multimodal_config.get('input_format', 'single_table') == 'single_table':
            config = deepcopy(config)
            config.setdefault('data', {}).setdefault('multimodal', {})['path'] = data_dir

        builder = MultimodalProcessedDataBuilder(config)
        multi_source_data, single_source_data = builder.build()

        output_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
        os.makedirs(output_dir, exist_ok=True)

        multi_path = os.path.join(output_dir, 'multi_source_data.pkl')
        with open(multi_path, 'wb') as f:
            pickle.dump(multi_source_data, f)
        logger.info(f"多模态多源数据已保存: {multi_path}")

        single_path = os.path.join(output_dir, 'single_source_data.pkl')
        with open(single_path, 'wb') as f:
            pickle.dump(single_source_data, f)
        logger.info(f"多模态单源基线数据已保存: {single_path}")

        return multi_source_data, single_source_data

    from src.data.dataloader import (
        CICIDS2017Preprocessor, MultiSourceDataSplitter, DataSplitter, ThreatIntelFeatureBuilder
    )

    # 1. 加载和预处理数据
    preprocessor = CICIDS2017Preprocessor(config)

    # 预处理配置
    preprocess_config = config.get('data', {}).get('preprocessing', {})

    # 支持多机构数据源（路径可在 --data_dir 和 config.data.institutional_sources 中同时配置）
    source_specs = _resolve_institutional_sources(data_dir, config, logger)
    if not source_specs:
        raise ValueError("没有可用的数据源，请检查 --data_dir 或 config.data.institutional_sources")

    logger.info("启用数据源: " + ", ".join(
        f"{item['name']}({item.get('institution', 'local')})" for item in source_specs
    ))

    if len(source_specs) == 1:
        logger.info(f"数据目录: {source_specs[0]['path']}")
        result = preprocessor.preprocess(
            data_path=source_specs[0]['path'],
            binary_classification=preprocess_config.get('binary_classification', False),
            feature_selection=preprocess_config.get('feature_selection', 'correlation'),
            normalize=preprocess_config.get('normalize', True)
        )
    else:
        logger.info(f"检测到 {len(source_specs)} 个机构数据源，开始合并预处理...")
        data_frames = []
        total_files = 0
        for source in source_specs:
            source_path = source['path']
            if os.path.isdir(source_path):
                file_paths = sorted(glob.glob(os.path.join(source_path, "*.csv")))
                if not file_paths:
                    logger.warning(f"目录中未找到CSV文件: {source_path}")
                    continue
                df = preprocessor.load_multiple_files(file_paths)
                total_files += len(file_paths)
            else:
                df = preprocessor.load_single_file(source_path)
                total_files += 1

            df['__institution_source__'] = source.get('name', 'unknown')
            data_frames.append(df)

        if not data_frames:
            raise ValueError("未从任何机构数据源中读取到CSV数据")

        import pandas as pd
        _validate_tabular_source_schema(data_frames, source_specs)
        merged_df = pd.concat(data_frames, ignore_index=True)
        if '__institution_source__' in merged_df.columns:
            merged_df = merged_df.drop(columns=['__institution_source__'])

        logger.info(f"合并完成: {len(data_frames)} 个数据源, {total_files} 个CSV文件, 合并后 {merged_df.shape[0]} 条样本")
        result = preprocessor.preprocess_dataframe(
            df=merged_df,
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
    _validate_multi_source_groups(ms_config, splitter, logger)

    X1, X2, names1, names2 = splitter.split(X, feature_names)
    source_arrays = [X1, X2]
    source_feature_names = [names1, names2]
    source_aliases = [
        '+'.join(ms_config.get('source1_groups', ['traffic', 'temporal'])),
        '+'.join(ms_config.get('source2_groups', ['flags', 'header', 'bulk']))
    ]

    extra_source_groups = ms_config.get('extra_source_groups', [])
    for idx, groups in enumerate(extra_source_groups, start=3):
        group_list = [groups] if isinstance(groups, str) else list(groups)
        feature_indices = splitter.get_feature_indices(feature_names, group_list)
        if not feature_indices:
            logger.warning(f"额外数据源 {idx} 未匹配到特征组: {group_list}，已跳过")
            continue
        source_arrays.append(X[:, feature_indices])
        source_feature_names.append([feature_names[i] for i in feature_indices])
        source_aliases.append('+'.join(group_list))

    threat_intel_config = config.get('data', {}).get('threat_intel', {})
    threat_intel_metadata = None
    if threat_intel_config.get('enabled', False):
        logger.info("加载威胁情报特征数据源...")
        intel_builder = ThreatIntelFeatureBuilder(threat_intel_config)
        intel_features, intel_feature_names = intel_builder.build_features(X.shape[0])
        source_arrays.append(intel_features)
        source_feature_names.append(intel_feature_names)
        source_aliases.append(intel_builder.source_name)
        threat_intel_metadata = {
            'enabled': True,
            'source_name': intel_builder.source_name,
            'source_path': threat_intel_config.get('source_path'),
            'join_strategy': threat_intel_config.get('join_strategy', 'row_order'),
            'feature_names': intel_feature_names,
        }
        logger.info(
            f"威胁情报数据源已接入: {intel_builder.source_name}, "
            f"{intel_features.shape[0]} samples, {intel_features.shape[1]} features"
        )

    logger.info(
        "数据源维度: " + ", ".join(
            f"源{i}:{arr.shape[1]}" for i, arr in enumerate(source_arrays, start=1)
        )
    )

    # 3. 数据集划分
    logger.info("划分训练/验证/测试集...")
    split_config = config.get('data', {}).get('split', {})
    data_splitter = DataSplitter(
        test_size=split_config.get('test_size', 0.2),
        val_size=split_config.get('val_size', 0.1),
        random_state=split_config.get('random_state', 42)
    )

    split_data = data_splitter.split_multi_source_list(
        source_arrays,
        y,
        stratify=split_config.get('stratify', True)
    )

    logger.info(f"训练集: {len(split_data['y_train'])} 样本")
    logger.info(f"验证集: {len(split_data['y_val'])} 样本")
    logger.info(f"测试集: {len(split_data['y_test'])} 样本")

    # 4. 保存处理后的数据
    output_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    os.makedirs(output_dir, exist_ok=True)

    # 多源数据
    multi_source_data = {
        'y_train': split_data['y_train'],
        'y_val': split_data['y_val'],
        'y_test': split_data['y_test'],
        'class_names': class_names,
        'num_classes': len(class_names),
        'num_sources': len(source_arrays),
        'source_aliases': source_aliases,
        'institution_sources': [item.get('name', 'unknown') for item in source_specs]
    }
    for i, feature_names_i in enumerate(source_feature_names, start=1):
        multi_source_data[f's{i}_train'] = split_data[f'X{i}_train']
        multi_source_data[f's{i}_val'] = split_data[f'X{i}_val']
        multi_source_data[f's{i}_test'] = split_data[f'X{i}_test']
        multi_source_data[f'source{i}_names'] = feature_names_i
        multi_source_data[f'source{i}_dim'] = source_arrays[i - 1].shape[1]

    # 向后兼容字段
    multi_source_data['source1_names'] = source_feature_names[0]
    multi_source_data['source2_names'] = source_feature_names[1]
    multi_source_data['source1_dim'] = source_arrays[0].shape[1]
    multi_source_data['source2_dim'] = source_arrays[1].shape[1]
    if threat_intel_metadata is not None:
        multi_source_data['threat_intel'] = threat_intel_metadata

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

    data_dict, source_indices = _build_loader_data_dict(data)
    source_dims = _infer_source_dims(data, source_indices)
    source1_dim, source2_dim = source_dims[0], source_dims[1]
    logger.info("Source dims: " + ", ".join(str(dim) for dim in source_dims))

    logger.info(f"加载数据: {data_path}")
    logger.info(f"源1维度: {source1_dim}, 源2维度: {source2_dim}")
    if len(source_indices) > 2:
        logger.info(f"检测到 {len(source_indices)} 个数据源，训练时将源2..源N拼接后输入模型第二分支")
    logger.info(f"类别数: {data['num_classes']}")

    # 更新配置
    config['model']['source_dims'] = source_dims
    config['model']['source1_dim'] = source1_dim
    config['model']['source2_dim'] = source2_dim
    config['model']['num_classes'] = data['num_classes']

    loader_config = config.get('data', {}).get('loader', {})
    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=loader_config.get('batch_size', 64),
        num_workers=loader_config.get('num_workers', 4),
        pin_memory=loader_config.get('pin_memory', device.type == 'cuda'),
        use_weighted_sampler=loader_config.get('use_weighted_sampler', False),
        augment_train=loader_config.get('augment_train', False)
    )

    # 创建模型
    model_config, factory_config = _build_model_factory_config(config)

    model = create_model(
        model_type=model_config.get('type', 'fusion_net'),
        traffic_dim=source1_dim,
        log_dim=source2_dim,
        num_classes=data['num_classes'],
        config=factory_config
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
    from sklearn.metrics import classification_report

    from src.models.fusion_net import create_model
    from src.data.dataset import create_multi_source_loaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    processed_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    data_path = os.path.join(processed_dir, 'multi_source_data.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_dict, source_indices = _build_loader_data_dict(data)
    source_dims = _infer_source_dims(data, source_indices)
    source1_dim, source2_dim = source_dims[0], source_dims[1]
    loader_config = config.get('data', {}).get('loader', {})

    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=loader_config.get('batch_size', 64),
        num_workers=loader_config.get('num_workers', 0),
        pin_memory=loader_config.get('pin_memory', device.type == 'cuda'),
        use_weighted_sampler=False,
        augment_train=False
    )
    test_loader = loaders['test']

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
        traffic_dim=source1_dim,
        log_dim=source2_dim,
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

    # 使用综合评估器
    from src.evaluation.evaluator import ComprehensiveEvaluator

    output_dir = os.path.join(project_root, 'outputs', experiment_name)
    evaluator = ComprehensiveEvaluator(
        model=model,
        device=device,
        class_names=data['class_names'],
        output_dir=output_dir
    )

    eval_results = evaluator.evaluate(test_loader)
    evaluator.print_report(eval_results)

    # 分类报告
    logger.info("\n分类报告:")
    logger.info("\n" + classification_report(
        eval_results['predictions']['y_true'],
        eval_results['predictions']['y_pred'],
        labels=list(range(len(data['class_names']))),
        target_names=data['class_names'],
        zero_division=0
    ))

    # 保存兼容格式的结果（供可视化模块使用）
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'y_true': eval_results['predictions']['y_true'],
        'y_pred': eval_results['predictions']['y_pred'],
        'y_proba': eval_results['predictions']['y_proba'],
        'metrics': eval_results['basic_metrics'],
        'class_names': data['class_names'],
        'confidence_intervals': eval_results['confidence_intervals'],
        'per_class_metrics': eval_results['per_class_metrics'],
        'roc_data': eval_results['roc_data'],
        'pr_data': eval_results['pr_data'],
    }
    if eval_results['predictions'].get('attention_weights') is not None:
        results['attention_weights'] = eval_results['predictions']['attention_weights']
    if eval_results['predictions'].get('agentic_actions') is not None:
        results['agentic_actions'] = eval_results['predictions']['agentic_actions']
    if data.get('source_aliases'):
        results['source_names'] = data['source_aliases']

    results_path = os.path.join(results_dir, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"评估结果已保存: {results_path}")

    logger.info("模型评估完成!")
    return eval_results['basic_metrics']


def evaluate_model_fixed(config: dict, experiment_name: str, logger: logging.Logger):
    """Checkpoint-aware evaluation path used by the CLI."""
    import torch
    from sklearn.metrics import classification_report

    from src.models.fusion_net import create_model
    from src.data.dataset import create_multi_source_loaders
    from src.evaluation.evaluator import ComprehensiveEvaluator

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_dir = os.path.join(project_root, config.get('data', {}).get('processed_dir', 'data/processed'))
    data_path = os.path.join(processed_dir, 'multi_source_data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    data_dict, source_indices = _build_loader_data_dict(data)
    source_dims = _infer_source_dims(data, source_indices)
    source1_dim, source2_dim = source_dims[0], source_dims[1]

    if experiment_name is None:
        checkpoint_dir = os.path.join(project_root, 'outputs')
        if os.path.exists(checkpoint_dir):
            experiments = [
                d for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('exp_')
            ]
            if experiments:
                experiment_name = sorted(experiments)[-1]

    if experiment_name is None:
        logger.error("No trained experiment found")
        return None

    model_path = os.path.join(project_root, 'outputs', experiment_name, 'checkpoints', 'best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return None

    logger.info(f"Loading model checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_config = checkpoint.get('config')
    eval_config = _merge_config_dicts(config, checkpoint_config) if isinstance(checkpoint_config, dict) else config
    if isinstance(checkpoint_config, dict):
        logger.info("Using model architecture from checkpoint config for evaluation")
    eval_config.setdefault('model', {})
    eval_config['model']['source_dims'] = source_dims
    eval_config['model']['source1_dim'] = source1_dim
    eval_config['model']['source2_dim'] = source2_dim
    eval_config['model']['num_classes'] = data['num_classes']

    loader_config = eval_config.get('data', {}).get('loader', {})
    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=loader_config.get('batch_size', 64),
        num_workers=loader_config.get('num_workers', 0),
        pin_memory=loader_config.get('pin_memory', device.type == 'cuda'),
        use_weighted_sampler=False,
        augment_train=False
    )
    test_loader = loaders['test']

    model_config, factory_config = _build_model_factory_config(eval_config)
    model = create_model(
        model_type=model_config.get('type', 'fusion_net'),
        traffic_dim=source1_dim,
        log_dim=source2_dim,
        num_classes=data['num_classes'],
        config=factory_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    output_dir = os.path.join(project_root, 'outputs', experiment_name)
    evaluator = ComprehensiveEvaluator(
        model=model,
        device=device,
        class_names=data['class_names'],
        output_dir=output_dir
    )

    eval_results = evaluator.evaluate(test_loader)
    evaluator.print_report(eval_results)

    logger.info("\n鍒嗙被鎶ュ憡:")
    logger.info("\n" + classification_report(
        eval_results['predictions']['y_true'],
        eval_results['predictions']['y_pred'],
        labels=list(range(len(data['class_names']))),
        target_names=data['class_names'],
        zero_division=0
    ))

    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results = {
        'y_true': eval_results['predictions']['y_true'],
        'y_pred': eval_results['predictions']['y_pred'],
        'y_proba': eval_results['predictions']['y_proba'],
        'metrics': eval_results['basic_metrics'],
        'class_names': data['class_names'],
        'confidence_intervals': eval_results['confidence_intervals'],
        'per_class_metrics': eval_results['per_class_metrics'],
        'roc_data': eval_results['roc_data'],
        'pr_data': eval_results['pr_data'],
    }
    if eval_results['predictions'].get('attention_weights') is not None:
        results['attention_weights'] = eval_results['predictions']['attention_weights']
    if eval_results['predictions'].get('agentic_actions') is not None:
        results['agentic_actions'] = eval_results['predictions']['agentic_actions']
    if data.get('source_aliases'):
        results['source_names'] = data['source_aliases']

    results_path = os.path.join(results_dir, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"璇勪及缁撴灉宸蹭繚瀛? {results_path}")

    logger.info("妯″瀷璇勪及瀹屾垚!")
    return eval_results['basic_metrics']


evaluate_model = evaluate_model_fixed


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
    data_dict, source_indices = _build_loader_data_dict(data)
    source_dims = _infer_source_dims(data, source_indices)
    source1_dim, source2_dim = source_dims[0], source_dims[1]

    # 更新配置
    config['model']['source_dims'] = source_dims
    config['model']['source1_dim'] = source1_dim
    config['model']['source2_dim'] = source2_dim
    config['model']['num_classes'] = data['num_classes']
    if len(source_indices) > 2:
        logger.info(f"检测到 {len(source_indices)} 个数据源，消融实验将源2..源N拼接后输入模型")

    loaders = create_multi_source_loaders(
        data_dict,
        batch_size=config.get('data', {}).get('loader', {}).get('batch_size', 64),
        num_workers=config.get('data', {}).get('loader', {}).get('num_workers', 0),
        pin_memory=config.get('data', {}).get('loader', {}).get('pin_memory', device.type == 'cuda')
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
        help='主数据目录路径（可配合 config.data.institutional_sources 一起合并）'
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

            available_sources = _resolve_institutional_sources(data_dir, config, logger)
            if not available_sources:
                logger.error(f"主数据目录不可用: {data_dir}")
                logger.error("请使用 --data_dir 或 config.data.institutional_sources 指定至少一个可用数据源路径")
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
