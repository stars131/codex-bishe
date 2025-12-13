"""
可视化模块

提供数据分析、训练监控、模型评估和报告生成功能。
"""

from .plots import (
    PlotStyle,
    DataVisualizer,
    TrainingVisualizer,
    EvaluationVisualizer,
    AttentionVisualizer,
    COLOR_PALETTE,
    COLORS
)

from .monitor import (
    TrainingHistory,
    TensorBoardLogger,
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
    TrainingMonitor
)

from .report import (
    ExperimentReport,
    generate_full_report
)

__all__ = [
    # 绘图样式
    'PlotStyle',
    'COLOR_PALETTE',
    'COLORS',
    # 数据可视化
    'DataVisualizer',
    # 训练可视化
    'TrainingVisualizer',
    # 评估可视化
    'EvaluationVisualizer',
    # 注意力可视化
    'AttentionVisualizer',
    # 训练监控
    'TrainingHistory',
    'TensorBoardLogger',
    'EarlyStopping',
    'ModelCheckpoint',
    'ProgressBar',
    'TrainingMonitor',
    # 报告生成
    'ExperimentReport',
    'generate_full_report'
]
