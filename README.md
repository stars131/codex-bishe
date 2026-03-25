# 基于多源数据融合与 Agentic 决策的网络攻击检测系统

本项目面向毕业设计场景，当前已经整理为一套可直接在服务器运行的正式实验工程，核心目标是：

- 使用 `BCCC-CSE-CIC-IDS-2018` 数据集进行网络攻击检测实验
- 构建 `flow + log/context` 双分支特征融合
- 将威胁情报作为第三源，采用 `decision-level fusion`
- 在推理阶段加入 `Agentic` 决策控制
- 自动输出论文可用的表格、图像和实验报告

如果你现在的目标是“在服务器上一键跑完整实验并产出论文材料”，优先看：

- `run_bccc_formal_experiments.py`
- `server_formal_bccc_experiments.sh`
- `docs/bccc_cicids2018_agentic.md`

## 1. 当前项目能做什么

当前仓库已经支持以下两类工作流。

### A. 快速验证工作流

适合本地或服务器先抽小样本验证整条链路：

- 从 BCCC 双层压缩包中抽样
- 自动构造 `flow_*` 特征
- 自动派生 `log_*` 上下文特征
- 自动生成本地威胁情报库
- 启动模拟 HTTP API
- 调用 API 生成 threat-intel sidecar
- 执行预处理、训练、测试

入口：

- `python run_bccc_agentic_demo.py`

### B. 正式实验工作流

适合毕业设计/论文正式实验：

- 批量运行主实验、消融实验、融合方法对比实验
- 每个实验单独保存 checkpoint、metrics、图像和 HTML 报告
- 汇总输出 CSV 表、LaTeX 表、Markdown 总报告、对比图
- 支持服务器一键运行

入口：

- `python run_bccc_formal_experiments.py --suite full`
- `bash server_formal_bccc_experiments.sh`

## 2. 当前默认实验设计

正式实验默认包含这些实验组：

1. `flow_only_single`
2. `log_only_single`
3. `flow_log_attention`
4. `flow_log_gated`
5. `flow_log_multi_head`
6. `flow_log_ti_decision`
7. `flow_log_ti_agentic`

其中：

- `flow_only_single`：只使用流量特征
- `log_only_single`：只使用派生上下文/日志特征
- `flow_log_*`：两源特征融合对比
- `flow_log_ti_decision`：加入威胁情报的决策层融合
- `flow_log_ti_agentic`：在决策层融合基础上加入 Agentic 推理控制

## 3. 数据集要求

当前主线数据集为：

- `BCCC-CSE-CIC-IDS-2018`

默认本地路径：

- `data/数据集BCCC-CSE-CIC-IDS-2018`

这个数据集是双层压缩结构：

- 外层 zip：按日期划分
- 内层 zip：按攻击类别或 benign 划分
- 最内层 csv：真实流级样本

本项目已经实现自动读取这类结构，不要求你先手工全部解压。

## 4. 项目结构

```text
main.py
run_bccc_agentic_demo.py
run_bccc_formal_experiments.py
serve_mock_threat_intel_api.py
server_formal_bccc_experiments.sh

src/
  config/
  data/
  evaluation/
  experiments/
  models/
  threat_intel/
  visualization/

docs/
tests/
```

重点目录说明：

- `src/data/`：数据读取、BCCC 适配、多模态构造
- `src/models/`：单源、多源、决策层融合、Agentic 相关模型
- `src/threat_intel/`：本地威胁情报库与模拟 API
- `src/experiments/`：正式实验总控与论文级结果汇总
- `src/visualization/`：图像、报告和 Streamlit 页面
- `docs/`：使用说明
- `tests/`：核心回归测试

## 5. 安装环境

### 本地 / Linux / 服务器

```bash
pip install -r requirements.txt
```

主要依赖：

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `pytest`

## 6. 快速开始

### 6.1 小样本一键验证

```bash
python run_bccc_agentic_demo.py
```

可选参数：

```bash
python run_bccc_agentic_demo.py \
  --sample-per-member 120 \
  --max-members 6 \
  --epochs 2 \
  --batch-size 64
```

这个脚本会自动完成：

1. 从 BCCC 压缩包抽样
2. 生成 `flow` / `log` 多模态文件
3. 生成 mock threat-intel library
4. 启动 mock API
5. 调用 API 得到 threat-intel sidecar
6. 预处理
7. 训练
8. 测试

## 7. 正式实验

### 7.1 运行 quick 套件

适合先验证服务器和代码环境。

```bash
python run_bccc_formal_experiments.py --suite quick
```

### 7.2 运行 full 套件

适合论文正式实验。

```bash
python run_bccc_formal_experiments.py --suite full
```

### 7.3 全量/大样本建议

如果要尽量接近正式论文实验，建议服务器上使用类似参数：

```bash
python run_bccc_formal_experiments.py \
  --suite full \
  --sample-per-member 0 \
  --max-members 0 \
  --epochs 30 \
  --batch-size 256 \
  --member-keywords ""
```

参数说明：

- `--sample-per-member 0`：每个内层 zip 全量读取
- `--max-members 0`：不限制成员数量
- `--member-keywords ""`：不按关键词筛选攻击包

### 7.4 自定义只跑部分实验

```bash
python run_bccc_formal_experiments.py \
  --experiments flow_log_attention,flow_log_ti_decision,flow_log_ti_agentic
```

## 8. 服务器一键运行

### 8.1 正式实验推荐入口

```bash
bash server_formal_bccc_experiments.sh
```

这个脚本会自动：

1. 检查 Python / GPU
2. 安装依赖
3. 调用正式实验总控

你也可以通过环境变量覆盖参数：

```bash
DATASET_DIR=data/数据集BCCC-CSE-CIC-IDS-2018 \
SUITE=full \
SAMPLE_PER_MEMBER=0 \
MAX_MEMBERS=0 \
EPOCHS=30 \
BATCH_SIZE=256 \
bash server_formal_bccc_experiments.sh
```

### 8.2 小样本 demo 服务器入口

```bash
bash server_train_bccc_agentic.sh
```

## 9. 威胁情报设计

本项目中的威胁情报模块不是直接把当前样本标签写回特征，而是走下面这条路径：

1. 根据样本构建本地 IOC 库
2. 启动模拟 HTTP API
3. 通过 API 查询 `src_ip / dst_ip / src_port / dst_port / protocol`
4. 返回数值化情报特征
5. 作为第三源参与决策层融合

默认实现位置：

- `src/threat_intel/mock_api.py`

单独启动 API：

```bash
python serve_mock_threat_intel_api.py \
  --library data/threat_intel/bccc_cicids2018_mock_library.json
```

## 10. 输出内容

正式实验运行后，会在 `outputs/formal_bccc_<timestamp>/` 下生成：

- `tables/experiment_summary.csv`
- `tables/experiment_summary.tex`
- `tables/per_class_metrics.csv`
- `tables/per_class_metrics.tex`
- `tables/confidence_intervals.csv`
- `tables/mcnemar_flow_log_attention_vs_agentic.json`
- `comparison_figures/metric_comparison.png`
- `comparison_figures/per_class_f1_heatmap.png`
- `comparison_figures/attention_comparison.png`
- `comparison_figures/validation_macro_f1_curves.png`
- `comparison_figures/agentic_action_distribution.png`
- `dataset_report/.../report.html`
- `formal_experiment_report.md`

每个子实验目录还会包含：

- `checkpoints/best_model.pth`
- `results/evaluation_metrics.json`
- `results/test_results.pkl`
- `figures/`
- `reports/<experiment>/report.html`

这些文件可以直接用于毕业设计论文中的：

- 实验结果表
- 分类性能对比图
- 消融实验图
- 注意力可视化图
- 训练过程图
- 数据集统计图

## 11. Web 页面

仓库保留了 Streamlit 可视化页面：

```bash
streamlit run src/visualization/app.py
```

但当前最稳定、最完整的正式实验入口仍然是脚本方式：

- `run_bccc_formal_experiments.py`
- `server_formal_bccc_experiments.sh`

如果你后续需要“网页上一键启动正式实验”，可以在这个基础上继续扩展。

## 12. 重要说明

### 12.1 关于 log 分支

BCCC 原始数据本身是流级 CSV，不是主机日志文件。  
为了满足“多源数据融合”的毕业设计目标，当前项目从以下字段派生出第二分支的 `log/context` 特征：

- 时间戳
- 端口
- 协议
- 握手状态
- 握手完整性标记
- IP 网络上下文

这是一种可复现实验设计，适合论文实现与对比实验。

### 12.2 关于正式结果

小样本 quick/demo 结果只用于：

- 验证代码链路
- 验证服务器环境
- 验证实验输出是否完整

论文最终结果应以更大样本或全量正式实验为准。

## 13. 相关脚本速查

- `run_bccc_agentic_demo.py`：BCCC 小样本一键 demo
- `run_bccc_formal_experiments.py`：BCCC 正式实验总控
- `serve_mock_threat_intel_api.py`：单独启动 mock threat-intel API
- `server_train_bccc_agentic.sh`：服务器 demo 入口
- `server_formal_bccc_experiments.sh`：服务器正式实验入口
- `run_um_nids_agentic.py`：UM-NIDS 处理后文件的多模态接入入口

## 14. 文档

- `docs/bccc_cicids2018_agentic.md`
- `docs/cicids2018_agentic_data_format.md`
- `docs/project_intro.md`

## 15. 许可证与用途

本项目主要用于课程设计、毕业设计和学术研究。  
请不要将其直接作为生产环境安全产品使用。
