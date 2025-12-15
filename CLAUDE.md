# 项目记忆文件

## 项目概述

**项目名称**: 基于多源数据融合的网络攻击检测系统

### 主要内容

网络攻击检测是保护网络安全的重要技术，已经有很多研究工作并取得一定效果，传统的检测方法是基于特征的方法，近年来基于机器学习的攻击检测方法成为主流，但目前网络攻击检测仍然面临攻击多样化、数据集、基于机器学习检测方法适用性、安全性等方面的挑战。

本课题将聚焦基于多源数据融合的攻击检测方法，多源数据包括流量和日志等信息，基于融合后的数据通过对现有的基于深度学习检测方法的总结和分析，通过实验分析、理论分析等方法进行算法分析，进而设计一种新的基于深度学习检测方法，实现网络攻击检测能力的提升。

### 基本要求

在大量阅读文献的基础上，设计出一种基于多源数据融合的攻击检测方法，并编写实验代码，获得实验数据，完成毕业论文撰写。如有可能，可尝试申请专利或者发表论文。

### 现有条件

- 开发语言：Python
- 数据集：CIC-IDS-2017 (位于 `/mnt/hgfs/linux-desktop共享文件夹/数据/CIC-IDS-2017/`)
- 框架：PyTorch

## 时间安排

| 时间 | 任务 | 状态 |
|------|------|------|
| 2026年2-3月 | 收集资料、学习基础知识 | 待开始 |
| 2026年4月 | 编写程序，完成所有功能 | 待开始 |
| 2026年5月 | 对程序进行改进，并撰写论文 | 待开始 |
| 2026年6月 | 做好答辩准备 | 待开始 |

## 预期成果

1. **模拟系统** - 完整的网络攻击检测系统
2. **毕业论文** - 学术论文撰写

---

## 已完成功能 ✅

### 1. 数据处理模块 (`src/data/`)
- [x] `dataloader.py` - CIC-IDS-2017数据加载、清洗、特征选择、归一化
- [x] `dataset.py` - PyTorch Dataset实现、数据增强、DataLoader创建
- [x] `preprocess.py` - 完整预处理脚本
- [x] `visualization.py` - 数据分析可视化
- [x] `__init__.py` - 模块导出

### 2. 模型模块 (`src/models/`) ✨全面升级

#### 融合网络 (`fusion_net.py`)
- [x] **FusionNet** - 多源数据融合主网络
  - 支持多种编码器：MLP、CNN、LSTM、Transformer
  - 支持多种融合方法：注意力、多头注意力、交叉注意力、门控、双线性、拼接
  - 正确返回 (logits, attention_weights) 元组
- [x] **SingleSourceNet** - 单源基线模型
- [x] **EnsembleFusionNet** - 集成多种融合方法

#### 融合模块
- [x] `AttentionFusion` - 基础注意力融合
- [x] `MultiHeadAttentionFusion` - 多头注意力融合 ✨新增
- [x] `CrossAttentionFusion` - 交叉注意力融合 ✨新增
- [x] `GatedFusion` - 门控融合 ✨新增
- [x] `BilinearFusion` - 双线性融合 ✨新增

#### 编码器
- [x] `MLPEncoder` - 多层感知机编码器（带残差）✨新增
- [x] `CNNEncoder` - 多尺度1D卷积编码器 ✨新增
- [x] `LSTMEncoder` - 双向LSTM编码器 ✨新增
- [x] `TransformerEncoder` - Transformer编码器 ✨新增

#### 损失函数 (`losses.py`) ✨新增
- [x] `FocalLoss` - 处理类不平衡
- [x] `LabelSmoothingCrossEntropy` - 标签平滑
- [x] `AsymmetricLoss` - 非对称损失
- [x] `DiceLoss` - Dice损失
- [x] `ClassBalancedLoss` - 基于有效样本数的类平衡损失
- [x] `ContrastiveLoss` - 对比学习损失
- [x] `CenterLoss` - 中心损失
- [x] `create_loss_function()` - 损失函数工厂

#### 可解释性模块 (`interpretability.py`) ✨新增
- [x] `AttentionAnalyzer` - 注意力权重分析
- [x] `FeatureImportanceAnalyzer` - 特征重要性分析（置换重要性、梯度重要性）
- [x] `GradCAM` - 梯度加权类激活映射
- [x] `ModelExplainer` - 综合模型解释器

### 3. 训练模块 (`src/train.py`) ✨全面重写

- [x] **Trainer类** - 高级训练器
  - 混合精度训练 (AMP)
  - 梯度累积
  - 学习率预热 (Warmup)
  - 早停机制
  - 模型检查点
  - TensorBoard日志
- [x] **WarmupScheduler** - 学习率预热调度器
- [x] **AblationStudy** - 消融实验管理器
  - 融合方法对比
  - 编码器对比

### 4. 工具模块 (`src/utils/helpers.py`) ✨全面升级

- [x] 配置管理：`load_config`, `save_config`, `merge_configs`
- [x] 检查点管理：`save_checkpoint`, `load_checkpoint`, `get_latest_checkpoint`
- [x] 评估指标：`evaluate_model`, `compute_roc_curve`, `compute_pr_curve`
- [x] 日志工具：`setup_logger`, `TensorBoardLogger`
- [x] 模型工具：`count_parameters`, `get_model_size`, `freeze_layers`
- [x] 时间工具：`Timer`, `format_time`
- [x] 数据工具：`save_results`, `load_results`

### 5. 可视化模块 (`src/visualization/`)
- [x] `app.py` - Streamlit交互式仪表板
- [x] `plots.py` - Matplotlib绑图工具
- [x] `monitor.py` - 训练监控
- [x] `report.py` - HTML报告生成

### 6. 配置模块 (`src/config/`)
- [x] `config.yaml` - 完整配置

### 7. 主入口 (`main.py`) ✨全面重写
- [x] 支持模式：full/preprocess/train/evaluate/report/dashboard/ablation
- [x] 完整的日志系统
- [x] 错误处理和异常追踪
- [x] 与数据加载模块完全一致

### 8. 部署
- [x] `deploy.sh` - Docker一键部署脚本
- [x] `Dockerfile` - Docker镜像配置
- [x] `docker-compose.yml` - 多服务编排

---

## 待完成功能 ⏳

### 1. 训练和实验
- [ ] 使用完整CIC-IDS-2017数据集训练模型
- [ ] 超参数调优（使用Optuna）
- [ ] 生成完整实验报告

### 2. 可视化增强
- [ ] PDF报告导出
- [ ] 实时预测界面

### 3. 论文撰写
- [ ] 文献综述
- [ ] 方法论描述
- [ ] 实验结果分析
- [ ] 结论与展望

---

## 使用方法

### 本地运行
```bash
# 安装依赖
pip install -r requirements.txt

# 完整流程（预处理 + 训练 + 评估 + 报告）
python main.py --data_dir "/path/to/CIC-IDS-2017" --mode full

# 仅预处理数据
python main.py --data_dir "/path/to/CIC-IDS-2017" --mode preprocess

# 仅训练模型
python main.py --mode train

# 仅评估模型
python main.py --mode evaluate

# 消融实验
python main.py --mode ablation

# 启动仪表板
python main.py --mode dashboard

# 使用独立训练脚本（带消融实验）
python src/train.py --config src/config/config.yaml --ablation
```

### Docker部署
```bash
# 一键部署
sudo ./deploy.sh

# 访问仪表板: http://localhost:8501
```

---

## 模型架构说明

### FusionNet 参数
```python
model = FusionNet(
    traffic_dim=50,          # 流量特征维度
    log_dim=30,              # 日志特征维度
    hidden_dim=256,          # 隐藏层维度
    num_classes=15,          # 类别数
    dropout=0.3,             # Dropout比率
    encoder_type='mlp',      # 编码器类型: mlp/cnn/lstm/transformer
    fusion_type='attention', # 融合类型: attention/multi_head/cross/gated/bilinear/concat
    num_layers=2,            # 编码器层数
    num_heads=4              # 注意力头数
)
```

### 支持的损失函数
```python
# 使用工厂函数创建
criterion = create_loss_function(
    loss_type='focal',       # cross_entropy/focal/label_smoothing/asymmetric/dice/class_balanced
    num_classes=15,
    class_weights=[...],     # 可选：类别权重
    gamma=2.0,               # Focal Loss gamma
    label_smoothing=0.1      # 标签平滑系数
)
```

---

## 项目结构

```
huanjing/
├── main.py                    # 主入口脚本
├── src/
│   ├── data/
│   │   ├── dataloader.py      # 数据加载和预处理
│   │   ├── dataset.py         # PyTorch Dataset
│   │   └── visualization.py   # 数据可视化
│   ├── models/
│   │   ├── fusion_net.py      # 融合网络和编码器
│   │   ├── losses.py          # 损失函数
│   │   └── interpretability.py # 可解释性模块
│   ├── train.py               # 训练脚本
│   ├── utils/
│   │   └── helpers.py         # 工具函数
│   ├── visualization/
│   │   ├── app.py             # Streamlit仪表板
│   │   ├── plots.py           # 绑图工具
│   │   ├── monitor.py         # 训练监控
│   │   └── report.py          # 报告生成
│   └── config/
│       └── config.yaml        # 配置文件
├── data/
│   ├── raw/                   # 原始数据
│   └── processed/             # 预处理后数据
├── outputs/
│   ├── checkpoints/           # 模型检查点
│   ├── results/               # 实验结果
│   ├── logs/                  # 日志文件
│   └── reports/               # 实验报告
├── deploy.sh                  # 部署脚本
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── CLAUDE.md                  # 项目记忆文件
```

---

## 当前进展 (2025-12-15)

### 已完成测试
- [x] Streamlit可视化仪表板启动成功 (http://localhost:8501)
- [x] 代码提交到git (commit: 5128207)

### 待处理
- [ ] **代码审计** - 检查各模块是否有bug或需要修改的地方
- [ ] 数据预处理测试
- [ ] 模型训练测试
- [ ] 消融实验测试

### 已知问题
- Docker权限问题：用户不在docker组，需要 `sudo usermod -aG docker $USER` 后重新登录
- data/processed 目录为空，需先运行预处理

---

## 更新日志

### v2.0 (2025-12-13)
- ✨ 重写模型模块，添加多种编码器和融合方法
- ✨ 添加Focal Loss等多种损失函数
- ✨ 重写训练脚本，支持混合精度、梯度累积、学习率预热
- ✨ 添加消融实验功能
- ✨ 添加模型可解释性模块
- ✨ 重写main.py，修复与数据模块的一致性问题
- ✨ 完善工具函数和日志系统

### v1.0
- 初始版本
- 基础数据处理和模型架构
- Streamlit可视化仪表板
