"""
实验报告生成器

自动生成完整的实验报告，包含数据分析、训练过程、模型评估等内容。
支持生成HTML和图片格式的报告。
"""
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 导入可视化模块
from .plots import (
    DataVisualizer, TrainingVisualizer, EvaluationVisualizer, AttentionVisualizer
)


class ExperimentReport:
    """
    实验报告生成器

    整合所有可视化结果，生成完整的实验报告。
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "outputs/reports",
        figures_dir: str = "outputs/figures"
    ):
        """
        初始化报告生成器

        Args:
            experiment_name: 实验名称
            output_dir: 报告输出目录
            figures_dir: 图片保存目录
        """
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(output_dir, experiment_name)
        self.figures_dir = figures_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # 初始化可视化器
        self.data_viz = DataVisualizer(os.path.join(figures_dir, "data"))
        self.train_viz = TrainingVisualizer(os.path.join(figures_dir, "training"))
        self.eval_viz = EvaluationVisualizer(os.path.join(figures_dir, "evaluation"))
        self.attn_viz = AttentionVisualizer(os.path.join(figures_dir, "attention"))

        # 存储报告数据
        self.report_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sections': {}
        }

    def add_data_analysis(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        class_names: List[str]
    ):
        """
        添加数据分析部分

        Args:
            features: 特征数组
            labels: 标签数组
            feature_names: 特征名称
            class_names: 类别名称
        """
        print("生成数据分析报告...")

        figures = {}

        # 1. 类别分布
        fig = self.data_viz.plot_class_distribution(
            labels, class_names,
            title="攻击类型分布",
            save_name="class_distribution.png"
        )
        figures['class_distribution'] = fig
        plt.close(fig)

        # 2. 相关性矩阵
        fig = self.data_viz.plot_correlation_matrix(
            features, feature_names,
            top_n=25,
            save_name="correlation_matrix.png"
        )
        figures['correlation_matrix'] = fig
        plt.close(fig)

        # 3. 特征分布
        fig = self.data_viz.plot_feature_distribution(
            features, labels, feature_names, class_names,
            save_name="feature_distribution.png"
        )
        figures['feature_distribution'] = fig
        plt.close(fig)

        # 4. 箱线图
        fig = self.data_viz.plot_boxplot_by_class(
            features, labels, feature_names, class_names,
            save_name="feature_boxplot.png"
        )
        figures['boxplot'] = fig
        plt.close(fig)

        # 5. 数据质量报告
        fig = self.data_viz.plot_data_quality_report(
            features, feature_names,
            save_name="data_quality.png"
        )
        figures['data_quality'] = fig
        plt.close(fig)

        # 6. 降维可视化 (可选)
        try:
            fig = self.data_viz.plot_dimensionality_reduction(
                features, labels, class_names,
                method='tsne',
                n_samples=3000,
                save_name="tsne_visualization.png"
            )
            figures['tsne'] = fig
            plt.close(fig)
        except Exception as e:
            print(f"t-SNE 可视化跳过: {e}")

        # 统计信息
        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            'num_samples': len(labels),
            'num_features': len(feature_names),
            'num_classes': len(class_names),
            'class_distribution': dict(zip([class_names[i] for i in unique], counts.tolist()))
        }

        self.report_data['sections']['data_analysis'] = {
            'figures': list(figures.keys()),
            'statistics': stats
        }

        print("数据分析完成!")

    def add_training_results(
        self,
        history: Dict[str, List[float]],
        config: Dict = None
    ):
        """
        添加训练结果部分

        Args:
            history: 训练历史字典
            config: 训练配置
        """
        print("生成训练过程报告...")

        figures = {}

        # 1. 训练曲线
        fig = self.train_viz.plot_training_curves(
            history,
            title="训练过程",
            save_name="training_curves.png"
        )
        figures['training_curves'] = fig
        plt.close(fig)

        # 2. 学习率曲线
        if 'learning_rate' in history:
            fig = self.train_viz.plot_learning_rate(
                history['learning_rate'],
                save_name="learning_rate.png"
            )
            figures['learning_rate'] = fig
            plt.close(fig)

        # 训练统计
        stats = {
            'total_epochs': len(history.get('train_loss', [])),
            'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
            'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
            'best_val_loss': min(history.get('val_loss', [float('inf')])),
            'best_val_acc': max(history.get('val_acc', [0])),
            'config': config
        }

        self.report_data['sections']['training'] = {
            'figures': list(figures.keys()),
            'statistics': stats
        }

        print("训练报告完成!")

    def add_evaluation_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        metrics: Dict[str, float] = None
    ):
        """
        添加评估结果部分

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            class_names: 类别名称
            metrics: 评估指标
        """
        print("生成评估结果报告...")

        figures = {}

        # 1. 混淆矩阵
        fig = self.eval_viz.plot_confusion_matrix(
            y_true, y_pred, class_names,
            normalize=True,
            save_name="confusion_matrix.png"
        )
        figures['confusion_matrix'] = fig
        plt.close(fig)

        # 非归一化混淆矩阵
        fig = self.eval_viz.plot_confusion_matrix(
            y_true, y_pred, class_names,
            normalize=False,
            title="混淆矩阵 (原始计数)",
            save_name="confusion_matrix_raw.png"
        )
        figures['confusion_matrix_raw'] = fig
        plt.close(fig)

        # 2. ROC曲线
        try:
            fig = self.eval_viz.plot_roc_curves(
                y_true, y_proba, class_names,
                save_name="roc_curves.png"
            )
            figures['roc_curves'] = fig
            plt.close(fig)
        except Exception as e:
            print(f"ROC曲线跳过: {e}")

        # 3. PR曲线
        try:
            fig = self.eval_viz.plot_precision_recall_curves(
                y_true, y_proba, class_names,
                save_name="pr_curves.png"
            )
            figures['pr_curves'] = fig
            plt.close(fig)
        except Exception as e:
            print(f"PR曲线跳过: {e}")

        # 4. 每类指标
        fig = self.eval_viz.plot_per_class_metrics(
            y_true, y_pred, class_names,
            save_name="per_class_metrics.png"
        )
        figures['per_class_metrics'] = fig
        plt.close(fig)

        self.report_data['sections']['evaluation'] = {
            'figures': list(figures.keys()),
            'metrics': metrics
        }

        print("评估报告完成!")

    def add_attention_analysis(
        self,
        attention_weights: np.ndarray,
        labels: np.ndarray = None,
        class_names: List[str] = None,
        source_names: List[str] = None
    ):
        """
        添加注意力分析部分

        Args:
            attention_weights: 注意力权重数组
            labels: 标签数组（用于按类别分析）
            class_names: 类别名称
            source_names: 数据源名称
        """
        print("生成注意力分析报告...")

        if source_names is None:
            source_names = ['流量+时序特征', '标志位+头部特征']

        figures = {}

        # 1. 注意力权重分布
        fig = self.attn_viz.plot_attention_weights(
            attention_weights, source_names,
            save_name="attention_weights.png"
        )
        figures['attention_weights'] = fig
        plt.close(fig)

        # 2. 按类别的注意力分析
        if labels is not None and class_names is not None:
            fig = self.attn_viz.plot_attention_by_class(
                attention_weights, labels, class_names, source_names,
                save_name="attention_by_class.png"
            )
            figures['attention_by_class'] = fig
            plt.close(fig)

        # 统计
        stats = {
            'mean_weights': np.mean(attention_weights, axis=0).tolist(),
            'std_weights': np.std(attention_weights, axis=0).tolist(),
            'source_names': source_names
        }

        self.report_data['sections']['attention'] = {
            'figures': list(figures.keys()),
            'statistics': stats
        }

        print("注意力分析完成!")

    def add_model_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metric_names: List[str] = None
    ):
        """
        添加模型对比部分

        Args:
            comparison_results: {模型名: {指标名: 值}} 的字典
            metric_names: 要对比的指标名列表
        """
        print("生成模型对比报告...")

        if metric_names is None:
            metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        figures = {}

        # 性能对比图
        fig = self.eval_viz.plot_metrics_comparison(
            comparison_results, metric_names,
            save_name="model_comparison.png"
        )
        figures['model_comparison'] = fig
        plt.close(fig)

        self.report_data['sections']['comparison'] = {
            'figures': list(figures.keys()),
            'results': comparison_results
        }

        print("模型对比完成!")

    def generate_html_report(self) -> str:
        """
        生成HTML格式的报告

        Returns:
            HTML报告的路径
        """
        print("生成HTML报告...")

        def _fmt(val, fmt='.4f'):
            """安全格式化数值"""
            try:
                return f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                return str(val) if val is not None else 'N/A'

        # 计算从报告目录到figures目录的相对路径
        figures_rel_path = os.path.relpath(self.figures_dir, self.output_dir)

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验报告 - {self.experiment_name}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .figure-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-card .label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ 网络攻击检测实验报告</h1>
        <p>实验名称: {self.experiment_name}</p>
        <p>生成时间: {self.report_data['timestamp']}</p>
    </div>
"""

        # 数据分析部分
        if 'data_analysis' in self.report_data['sections']:
            section = self.report_data['sections']['data_analysis']
            stats = section.get('statistics', {})
            html_content += f"""
    <div class="section">
        <h2>📊 数据分析</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{stats.get('num_samples', 'N/A'):,}</div>
                <div class="label">样本总数</div>
            </div>
            <div class="metric-card">
                <div class="value">{stats.get('num_features', 'N/A')}</div>
                <div class="label">特征维度</div>
            </div>
            <div class="metric-card">
                <div class="value">{stats.get('num_classes', 'N/A')}</div>
                <div class="label">类别数量</div>
            </div>
        </div>
        <div class="figure-container">
            <img src="{figures_rel_path}/data/class_distribution.png" alt="类别分布">
            <p>图1: 攻击类型分布</p>
        </div>
        <div class="figure-container">
            <img src="{figures_rel_path}/data/correlation_matrix.png" alt="相关性矩阵">
            <p>图2: 特征相关性矩阵</p>
        </div>
        <div class="figure-container">
            <img src="{figures_rel_path}/data/data_quality.png" alt="数据质量">
            <p>图3: 数据质量报告</p>
        </div>
    </div>
"""

        # 训练结果部分
        if 'training' in self.report_data['sections']:
            section = self.report_data['sections']['training']
            stats = section.get('statistics', {})
            html_content += f"""
    <div class="section">
        <h2>📈 训练过程</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{stats.get('total_epochs', 'N/A')}</div>
                <div class="label">训练轮数</div>
            </div>
            <div class="metric-card">
                <div class="value">{_fmt(stats.get('best_val_loss'))}</div>
                <div class="label">最佳验证损失</div>
            </div>
            <div class="metric-card">
                <div class="value">{_fmt(stats.get('best_val_acc'))}</div>
                <div class="label">最佳验证准确率</div>
            </div>
        </div>
        <div class="figure-container">
            <img src="{figures_rel_path}/training/training_curves.png" alt="训练曲线">
            <p>图4: 训练过程曲线</p>
        </div>
    </div>
"""

        # 评估结果部分
        if 'evaluation' in self.report_data['sections']:
            section = self.report_data['sections']['evaluation']
            metrics = section.get('metrics', {})
            html_content += f"""
    <div class="section">
        <h2>🎯 模型评估</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{_fmt(metrics.get('accuracy', 0))}</div>
                <div class="label">准确率</div>
            </div>
            <div class="metric-card">
                <div class="value">{_fmt(metrics.get('precision', metrics.get('precision_weighted', 0)))}</div>
                <div class="label">精确率</div>
            </div>
            <div class="metric-card">
                <div class="value">{_fmt(metrics.get('recall', metrics.get('recall_weighted', 0)))}</div>
                <div class="label">召回率</div>
            </div>
            <div class="metric-card">
                <div class="value">{_fmt(metrics.get('f1_score', metrics.get('f1_weighted', 0)))}</div>
                <div class="label">F1分数</div>
            </div>
        </div>
        <div class="figure-container">
            <img src="{figures_rel_path}/evaluation/confusion_matrix.png" alt="混淆矩阵">
            <p>图5: 混淆矩阵</p>
        </div>
        <div class="figure-container">
            <img src="{figures_rel_path}/evaluation/per_class_metrics.png" alt="各类别指标">
            <p>图6: 各类别性能指标</p>
        </div>
    </div>
"""

        # 注意力分析部分
        if 'attention' in self.report_data['sections']:
            section = self.report_data['sections']['attention']
            stats = section.get('statistics', {})
            html_content += f"""
    <div class="section">
        <h2>🔍 注意力分析</h2>
        <p>数据源: {', '.join(stats.get('source_names', []))}</p>
        <p>平均注意力权重: {', '.join(f'{w:.3f}' for w in stats.get('mean_weights', []))}</p>
        <div class="figure-container">
            <img src="{figures_rel_path}/attention/attention_weights.png" alt="注意力权重">
            <p>图7: 注意力权重分布</p>
        </div>
    </div>
"""

        # 模型对比部分
        if 'comparison' in self.report_data['sections']:
            section = self.report_data['sections']['comparison']
            html_content += f"""
    <div class="section">
        <h2>📝 模型对比</h2>
        <div class="figure-container">
            <img src="{figures_rel_path}/evaluation/model_comparison.png" alt="模型对比">
            <p>图8: 模型性能对比</p>
        </div>
    </div>
"""

        html_content += """
    <div class="footer">
        <p>基于多源数据融合的网络攻击检测系统 | 毕业设计项目</p>
        <p>报告自动生成</p>
    </div>
</body>
</html>
"""

        # 保存HTML
        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML报告已保存: {html_path}")
        return html_path

    def save_report_data(self):
        """保存报告数据"""
        data_path = os.path.join(self.output_dir, "report_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(self.report_data, f)
        print(f"报告数据已保存: {data_path}")


def generate_full_report(
    experiment_name: str,
    data_path: str = None,
    results_path: str = None,
    history_path: str = None,
    output_dir: str = "outputs/reports"
) -> str:
    """
    生成完整的实验报告

    Args:
        experiment_name: 实验名称
        data_path: 预处理数据路径
        results_path: 模型结果路径
        history_path: 训练历史路径
        output_dir: 输出目录

    Returns:
        报告路径
    """
    report = ExperimentReport(experiment_name, output_dir)

    # 加载数据
    if data_path and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        all_features = np.vstack([data['X_train'], data['X_val'], data['X_test']])
        all_labels = np.concatenate([data['y_train'], data['y_val'], data['y_test']])

        report.add_data_analysis(
            all_features, all_labels,
            data['feature_names'], data['class_names']
        )

    # 加载训练历史
    if history_path and os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history_data = pickle.load(f)
        report.add_training_results(history_data.get('history', history_data))

    # 加载评估结果
    if results_path and os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)

        report.add_evaluation_results(
            results['y_true'], results['y_pred'], results['y_proba'],
            results['class_names'], results.get('metrics')
        )

        if 'attention_weights' in results and results['attention_weights'] is not None:
            report.add_attention_analysis(
                results['attention_weights'],
                results['y_true'],
                results['class_names']
            )

    # 生成HTML报告
    html_path = report.generate_html_report()
    report.save_report_data()

    return html_path
