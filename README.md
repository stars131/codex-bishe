# 基于多源数据融合的网络攻击检测系统

本项目实现了一种基于多源数据融合的网络入侵检测方法，通过将网络流量特征分割为多个数据源，利用深度学习编码器独立提取特征后进行融合，实现对网络攻击的高效检测与分类。

## 系统架构

```
原始数据 → 预处理 → 多源特征分割 → 独立编码 → 特征融合 → 攻击分类
              ↓                        ↓           ↓
         数据清洗/归一化          MLP/CNN/LSTM   Attention/Gated
                                /Transformer    /Bilinear/Cross
```

### 核心模块

| 模块 | 说明 |
|------|------|
| **数据预处理** | 支持 CIC-IDS-2017 和 KDD Cup 99 数据集，自动清洗、编码、归一化 |
| **多源分割** | 将特征按类型分为两个数据源（流量/时序 + 标志位/头部/批量） |
| **融合模型** | FusionNet 支持 4 种编码器 × 6 种融合方法的灵活组合 |
| **训练引擎** | 支持混合精度训练、梯度累积、学习率预热、早停机制 |
| **消融实验** | 系统性对比不同编码器和融合方法的效果 |
| **可视化** | Streamlit 交互式仪表板，实时监控训练过程 |

## 项目结构

```
├── main.py                     # 主入口（CIC-IDS-2017 完整流程）
├── quick_test.py               # Windows 快速测试（KDD Cup 99）
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 镜像配置
├── docker-compose.yml          # Docker Compose 多服务编排
├── src/
│   ├── config/
│   │   ├── config.yaml         # 主配置文件
│   │   └── config_windows.yaml # Windows 专用配置
│   ├── data/
│   │   ├── dataloader.py       # CIC-IDS-2017 数据加载与预处理
│   │   ├── kddcup_loader.py    # KDD Cup 99 数据加载与预处理
│   │   ├── dataset.py          # PyTorch Dataset 定义
│   │   ├── preprocess.py       # 数据清洗与特征工程
│   │   └── visualization.py    # 数据探索可视化
│   ├── models/
│   │   ├── fusion_net.py       # 多源融合模型（FusionNet）
│   │   ├── losses.py           # 自定义损失函数
│   │   └── interpretability.py # 模型可解释性分析
│   ├── evaluation/
│   │   └── evaluator.py        # 综合评估器
│   ├── visualization/
│   │   ├── app.py              # Streamlit 仪表板
│   │   ├── plots.py            # 可视化绘图函数
│   │   ├── monitor.py          # 训练监控
│   │   └── report.py           # 报告生成
│   └── utils/
│       └── helpers.py          # 工具函数
├── tests/                      # 单元测试
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据集
│   └── processed/              # 预处理后数据
└── outputs/                    # 输出目录
    ├── checkpoints/            # 模型权重
    ├── results/                # 实验结果
    ├── figures/                # 可视化图表
    └── logs/                   # 训练日志
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- matplotlib / seaborn
- streamlit（可视化仪表板）

### Docker 部署

```bash
# 启动 Streamlit 仪表板
docker-compose up -d dashboard      # http://localhost:8501

# 运行训练
docker-compose run train

# 运行预处理
docker-compose run preprocess
```

## 使用方法

### 完整流程（CIC-IDS-2017）

```bash
# 预处理 + 训练 + 评估 + 报告
python main.py --data_dir "/path/to/CIC-IDS-2017" --mode full

# 单独运行各阶段
python main.py --data_dir "/path/to/CIC-IDS-2017" --mode preprocess
python main.py --mode train
python main.py --mode evaluate
python main.py --mode ablation
python main.py --mode dashboard
```

### Windows 快速测试（KDD Cup 99）

```bash
# 完整测试流程（自动下载数据、预处理、训练）
python quick_test.py

# 仅预处理 / 仅训练 / 启动仪表板
python quick_test.py --mode preprocess
python quick_test.py --mode train
python quick_test.py --mode dashboard

# 自定义参数
python quick_test.py --data_file "path/to/kddcup.csv" --epochs 50 --sample_size 100000
```

### 从检查点恢复训练

```bash
python src/train.py --resume outputs/checkpoints/best_model.pth
```

### 消融实验

```bash
python src/train.py --config src/config/config.yaml --ablation
```

## 模型说明

### 编码器类型

| 编码器 | 说明 |
|--------|------|
| `mlp` | 多层感知机，适用于结构化表格数据 |
| `cnn` | 一维卷积网络，捕获局部特征模式 |
| `lstm` | 长短时记忆网络，建模时序依赖关系 |
| `transformer` | 自注意力机制，捕获全局特征交互 |

### 融合方法

| 方法 | 说明 |
|------|------|
| `attention` | 注意力融合，自适应学习数据源权重 |
| `multi_head` | 多头注意力融合 |
| `cross` | 交叉注意力，建模数据源间交互 |
| `gated` | 门控融合，控制信息流通 |
| `bilinear` | 双线性融合，捕获二阶特征交互 |
| `concat` | 拼接融合，作为基线方法 |

### 损失函数

支持 `cross_entropy`、`focal`、`label_smoothing`、`asymmetric`、`dice`、`class_balanced` 等多种损失函数，可通过配置文件灵活切换。

## 配置说明

所有超参数集中在 `src/config/config.yaml` 中管理：

```yaml
model:
  architecture:
    hidden_dim: 256       # 隐藏层维度
    num_layers: 3         # 网络层数
    dropout: 0.3          # Dropout 比率
  fusion:
    method: "attention"   # 融合方法

training:
  epochs: 100
  optimizer:
    type: "adamw"
    learning_rate: 0.001
  early_stopping:
    patience: 15
```

## 支持的数据集

### CIC-IDS-2017
- 加拿大网络安全研究所发布的入侵检测数据集
- 包含正常流量和多种攻击类型（DDoS、PortScan、Brute Force 等）
- 多源分割：流量/时序特征 + 标志位/头部/批量特征

### KDD Cup 99
- 经典网络入侵检测基准数据集
- 5 分类：Normal、DoS、Probe、R2L、U2R
- 多源分割：基本连接/内容特征 + 流量/主机特征

## 评估指标

系统提供全面的评估体系：
- Accuracy、Precision、Recall、F1-Score
- ROC-AUC、PR-AUC
- 混淆矩阵
- 每类别详细指标
- Bootstrap 置信区间
- McNemar 统计检验

## 许可证

本项目仅供学术研究使用。
