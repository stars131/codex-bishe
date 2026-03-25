# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

基于多源数据融合的网络攻击检测系统。支持 CIC-IDS-2017 和 KDD Cup 99 数据集，通过 PyTorch 训练融合模型，结合流量特征和日志特征进行网络入侵检测。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 完整流程（预处理 + 训练 + 评估 + 报告）
python main.py --data_dir "/path/to/CIC-IDS-2017" --mode full

# 单独运行各模式
python main.py --data_dir "/path/to/CIC-IDS-2017" --mode preprocess
python main.py --mode train
python main.py --mode evaluate
python main.py --mode ablation
python main.py --mode dashboard

# 独立训练脚本（带消融实验）
python src/train.py --config src/config/config.yaml --ablation

# 从检查点恢复训练
python src/train.py --resume outputs/<experiment_name>/checkpoints/best_model.pth

# Docker 部署
docker-compose up -d dashboard      # 启动 Streamlit 仪表板 (http://localhost:8501)
docker-compose run train            # 运行训练
docker-compose run preprocess       # 运行预处理
```

## Windows 快速测试（KDD Cup 99）

使用 `quick_test.py` 在 Windows 系统上快速测试：

```bash
# 完整测试流程（自动下载数据、预处理、训练）
python quick_test.py

# 仅预处理数据
python quick_test.py --mode preprocess

# 仅训练模型
python quick_test.py --mode train

# 启动可视化仪表板
python quick_test.py --mode dashboard

# 指定数据文件和参数
python quick_test.py --data_file "path/to/kddcup.csv" --epochs 50 --sample_size 100000
```

### KDD Cup 数据处理

KDD Cup 99 预处理器位于 `src/data/kddcup_loader.py`：
- **KDDCupPreprocessor**: 数据加载、编码、标准化
- **KDDMultiSourceSplitter**: 多源数据分割
- 支持 5 分类（normal, dos, probe, r2l, u2r）或二分类

多源数据分割策略：
- **数据源1**: 基本连接特征 + 内容特征
- **数据源2**: 流量特征 + 主机特征

Windows 配置文件：`src/config/config_windows.yaml`（已设置 `num_workers: 0`）

## 架构说明

### 数据流程
1. **原始数据** (`data/raw/`) - CIC-IDS-2017 或 KDD Cup 99 CSV 文件
2. **预处理** (`src/data/dataloader.py` 或 `src/data/kddcup_loader.py`) - 加载、清洗、归一化、分割为多源特征
3. **数据集** (`src/data/dataset.py`) - PyTorch Dataset，使用 `create_multi_source_loaders()`
4. **模型** (`src/models/fusion_net.py`) - 编码各数据源、融合、分类
5. **训练** (`src/train.py`) - Trainer 类，支持 AMP、梯度累积、早停

### 多源数据分割
在 `src/config/config.yaml` 的 `data.multi_source` 中配置：
- **数据源1**: 流量特征 + 时序特征
- **数据源2**: 标志位特征 + 头部特征 + 批量特征

### 模型架构 (`src/models/fusion_net.py`)

**FusionNet** 接收两个特征张量，返回 `(logits, attention_weights)`：
- **编码器类型**: `mlp`, `cnn`, `lstm`, `transformer` - 独立编码各数据源
- **融合方法**: `attention`, `multi_head`, `cross`, `gated`, `bilinear`, `concat`
- **SingleSourceNet**: 单源基线模型，用于消融实验
- **EnsembleFusionNet**: 集成多种融合方法

关键类：
- `FusionNet(traffic_dim, log_dim, hidden_dim, num_classes, encoder_type, fusion_type)`
- `create_model(model_type, traffic_dim, log_dim, num_classes, config)` - 工厂函数

### 训练模块 (`src/train.py`)

**Trainer** 类功能：
- 混合精度训练 (AMP)
- 梯度累积
- 学习率预热 (`WarmupScheduler`)
- 早停机制
- 检查点保存至 `outputs/checkpoints/`

**AblationStudy** 类用于系统性对比不同融合方法和编码器。

### 损失函数 (`src/models/losses.py`)

使用 `create_loss_function(loss_type, num_classes, class_weights, gamma, label_smoothing)`：
- 支持类型: `cross_entropy`, `focal`, `label_smoothing`, `asymmetric`, `dice`, `class_balanced`

### 配置文件

所有超参数在 `src/config/config.yaml` 中：
- `data`: 路径、预处理选项、多源分割、数据加载器设置
- `model`: 模型类型、架构参数（hidden_dim, num_layers, dropout）、融合方法
- `training`: 训练轮数、优化器、调度器、早停、损失函数、梯度裁剪
- `ablation`: 消融实验配置

### 输出目录

- `outputs/checkpoints/` - 模型权重 (`best_model.pth`, `last_model.pth`)
- `outputs/results/` - 实验结果 JSON
- `outputs/<experiment_name>/train_*.log` - 训练日志
- `outputs/figures/` - 可视化图表
- `data/processed/multi_source_data.pkl` - 预处理数据缓存

## 关键模式

- 模型返回元组: `logits, attention_weights = model(source1, source2)`
- 多源数据加载器返回三元组: `(source1, source2, labels)`
- 配置以字典形式传递，使用 `src/utils/helpers.py` 中的 `load_config()` 加载
- 训练前必须先运行预处理，检查 `data/processed/multi_source_data.pkl` 是否存在
