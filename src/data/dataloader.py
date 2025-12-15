"""
Data loading and preprocessing module for network attack detection.
Supports CIC-IDS-2017 dataset with multi-source data fusion.
"""
import os
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# 可选导入 imbalanced-learn (兼容性处理)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("警告: imbalanced-learn 未安装或版本不兼容，数据平衡功能将不可用")

import warnings
warnings.filterwarnings('ignore')


class CICIDS2017Preprocessor:
    """
    CIC-IDS-2017 数据集专用预处理器

    功能：
    - 加载和合并多个CSV文件
    - 数据清洗（处理缺失值、无穷值、重复值）
    - 特征选择和工程
    - 标签编码（二分类/多分类）
    - 数据标准化
    """

    # CIC-IDS-2017 标准列名（共79个特征 + 1个标签）
    FEATURE_COLUMNS = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
        'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
        'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
        'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
        'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
        'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
        'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]

    LABEL_COLUMN = 'Label'

    # 攻击类型映射
    ATTACK_CATEGORIES = {
        'BENIGN': 'Benign',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'DDoS': 'DDoS',
        'PortScan': 'PortScan',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Bot',
        'Web Attack – Brute Force': 'Web Attack',
        'Web Attack – XSS': 'Web Attack',
        'Web Attack – Sql Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Heartbleed'
    }

    def __init__(self, config: Dict = None):
        """
        初始化预处理器

        Args:
            config: 配置字典，包含数据路径等信息
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.num_classes = None
        self.class_names = None

    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        加载单个CSV文件

        Args:
            file_path: CSV文件路径

        Returns:
            DataFrame
        """
        print(f"加载文件: {file_path}")

        # 尝试不同的编码方式
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"  - 成功使用 {encoding} 编码加载")
                print(f"  - 数据形状: {df.shape}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  - 加载错误: {e}")
                raise

        raise ValueError(f"无法加载文件: {file_path}")

    def load_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        加载并合并多个CSV文件

        Args:
            file_paths: CSV文件路径列表

        Returns:
            合并后的DataFrame
        """
        dfs = []
        for path in file_paths:
            if os.path.exists(path):
                df = self.load_single_file(path)
                dfs.append(df)
            else:
                print(f"警告: 文件不存在 - {path}")

        if not dfs:
            raise ValueError("没有成功加载任何数据文件")

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n合并后数据形状: {combined_df.shape}")
        return combined_df

    def load_from_directory(self, directory: str, pattern: str = "*.csv") -> pd.DataFrame:
        """
        从目录加载所有匹配的CSV文件

        Args:
            directory: 数据目录
            pattern: 文件匹配模式

        Returns:
            合并后的DataFrame
        """
        import glob
        file_paths = sorted(glob.glob(os.path.join(directory, pattern)))
        print(f"找到 {len(file_paths)} 个CSV文件")
        return self.load_multiple_files(file_paths)

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理列名（去除空格等）

        Args:
            df: 原始DataFrame

        Returns:
            清理后的DataFrame
        """
        df.columns = df.columns.str.strip()
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        处理：
        - 缺失值
        - 无穷值
        - 重复行
        - 异常值

        Args:
            df: 原始DataFrame

        Returns:
            清洗后的DataFrame
        """
        print("\n=== 数据清洗 ===")
        original_shape = df.shape

        # 1. 清理列名
        df = self.clean_column_names(df)

        # 2. 删除重复行
        df = df.drop_duplicates()
        print(f"删除重复行: {original_shape[0] - df.shape[0]} 行")

        # 3. 获取特征列（排除标签列）
        label_col = None
        for col in ['Label', ' Label', 'label']:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            raise ValueError("找不到标签列")

        feature_cols = [col for col in df.columns if col != label_col]

        # 4. 转换为数值类型
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. 处理无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        inf_count = df[feature_cols].isin([np.inf, -np.inf]).sum().sum()
        print(f"无穷值数量: {inf_count}")

        # 6. 处理缺失值
        missing_before = df[feature_cols].isnull().sum().sum()

        # 对于缺失比例较高的列，使用中位数填充
        for col in feature_cols:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0:
                df[col] = df[col].fillna(df[col].median())

        missing_after = df[feature_cols].isnull().sum().sum()
        print(f"缺失值: {missing_before} -> {missing_after}")

        # 7. 删除仍有缺失值的行
        df = df.dropna()

        print(f"最终数据形状: {df.shape}")
        return df

    def encode_labels(
        self,
        labels: pd.Series,
        binary: bool = False,
        fit: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        编码标签

        Args:
            labels: 标签Series
            binary: 是否二分类（正常 vs 攻击）
            fit: 是否fit encoder

        Returns:
            编码后的标签和类别名称
        """
        # 清理标签中的空格
        labels = labels.str.strip()

        if binary:
            # 二分类：Benign=0, Attack=1
            encoded = (labels != 'BENIGN').astype(int)
            class_names = ['Benign', 'Attack']
        else:
            # 多分类
            if fit:
                encoded = self.label_encoder.fit_transform(labels)
                class_names = list(self.label_encoder.classes_)
            else:
                encoded = self.label_encoder.transform(labels)
                class_names = list(self.label_encoder.classes_)

        self.num_classes = len(class_names)
        self.class_names = class_names

        return encoded, class_names

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = 'all',
        n_features: int = 50
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        特征选择

        Args:
            df: 数据DataFrame
            method: 选择方法 ('all', 'variance', 'correlation', 'importance')
            n_features: 选择的特征数量

        Returns:
            选择后的特征DataFrame和特征名列表
        """
        # 获取标签列
        label_col = None
        for col in ['Label', ' Label', 'label']:
            if col in df.columns:
                label_col = col
                break

        feature_cols = [col for col in df.columns if col != label_col]

        if method == 'all':
            selected_features = feature_cols
        elif method == 'variance':
            # 基于方差选择
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(df[feature_cols])
            selected_features = [f for f, s in zip(feature_cols, selector.get_support()) if s]
        elif method == 'correlation':
            # 移除高度相关的特征
            corr_matrix = df[feature_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            selected_features = [f for f in feature_cols if f not in to_drop]
        else:
            selected_features = feature_cols[:n_features]

        self.feature_names = selected_features
        print(f"\n选择了 {len(selected_features)} 个特征")

        return df[selected_features], selected_features

    def normalize_features(
        self,
        X: np.ndarray,
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """
        特征标准化

        Args:
            X: 特征数组
            method: 标准化方法 ('standard', 'minmax')
            fit: 是否fit scaler

        Returns:
            标准化后的特征
        """
        if method == 'minmax':
            if not hasattr(self, 'minmax_scaler'):
                self.minmax_scaler = MinMaxScaler()
            scaler = self.minmax_scaler
        else:
            scaler = self.scaler

        if fit:
            return scaler.fit_transform(X)
        return scaler.transform(X)

    def preprocess(
        self,
        data_path: Union[str, List[str]],
        binary_classification: bool = False,
        feature_selection: str = 'correlation',
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        完整的预处理流程

        Args:
            data_path: 数据文件路径或路径列表
            binary_classification: 是否二分类
            feature_selection: 特征选择方法
            normalize: 是否标准化
            save_path: 保存预处理结果的路径

        Returns:
            包含特征和标签的字典
        """
        print("=" * 50)
        print("CIC-IDS-2017 数据预处理")
        print("=" * 50)

        # 1. 加载数据
        if isinstance(data_path, list):
            df = self.load_multiple_files(data_path)
        elif os.path.isdir(data_path):
            df = self.load_from_directory(data_path)
        else:
            df = self.load_single_file(data_path)

        # 2. 数据清洗
        df = self.clean_data(df)

        # 3. 获取标签列
        label_col = None
        for col in ['Label', ' Label', 'label']:
            if col in df.columns:
                label_col = col
                break

        # 4. 编码标签
        y, class_names = self.encode_labels(df[label_col], binary=binary_classification)
        print(f"\n类别分布:")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {class_names[u]}: {c} ({c/len(y)*100:.2f}%)")

        # 5. 特征选择
        X_df, feature_names = self.select_features(df, method=feature_selection)
        X = X_df.values.astype(np.float32)

        # 6. 标准化
        if normalize:
            X = self.normalize_features(X)
            print("已完成特征标准化")

        result = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'class_names': class_names,
            'num_features': X.shape[1],
            'num_classes': len(class_names)
        }

        # 7. 保存结果
        if save_path:
            self.save_preprocessed(result, save_path)

        print("\n预处理完成!")
        print(f"特征维度: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"类别数量: {len(class_names)}")

        return result

    def save_preprocessed(self, data: Dict, path: str):
        """保存预处理后的数据"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"预处理数据已保存到: {path}")

        # 同时保存scaler和encoder
        scaler_path = path.replace('.pkl', '_scaler.pkl')
        encoder_path = path.replace('.pkl', '_encoder.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

    @staticmethod
    def load_preprocessed(path: str) -> Dict:
        """加载预处理后的数据"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class MultiSourceDataSplitter:
    """
    多源数据分割器

    将单一数据源的特征分割成多个"虚拟"数据源，
    用于模拟多源数据融合场景。

    分割策略：
    1. 流量统计特征 (traffic): 包大小、流量速率等
    2. 时序特征 (temporal): IAT、持续时间等
    3. 标志位特征 (flags): TCP标志位等
    4. 协议特征 (protocol): 端口、协议相关等
    """

    # 特征组定义
    FEATURE_GROUPS = {
        'traffic': [
            'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s',
            'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length',
            'Packet Length Mean', 'Packet Length Std',
            'Packet Length Variance', 'Average Packet Size',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes'
        ],
        'temporal': [
            'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
            'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
            'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ],
        'flags': [
            'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count'
        ],
        'header': [
            'Destination Port', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Header Length.1', 'Down/Up Ratio',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward'
        ],
        'bulk': [
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
            'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
            'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
        ]
    }

    def __init__(self, source1_groups: List[str] = None, source2_groups: List[str] = None):
        """
        初始化分割器

        Args:
            source1_groups: 第一个数据源包含的特征组
            source2_groups: 第二个数据源包含的特征组
        """
        # 默认分割方案：流量+时序 vs 标志位+头部
        self.source1_groups = source1_groups or ['traffic', 'temporal']
        self.source2_groups = source2_groups or ['flags', 'header', 'bulk']

    def get_feature_indices(
        self,
        feature_names: List[str],
        groups: List[str]
    ) -> List[int]:
        """获取指定特征组的特征索引"""
        target_features = []
        for group in groups:
            target_features.extend(self.FEATURE_GROUPS.get(group, []))

        indices = []
        for i, name in enumerate(feature_names):
            if name in target_features:
                indices.append(i)
        return indices

    def split(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        分割特征为两个数据源

        Args:
            X: 原始特征数组
            feature_names: 特征名称列表

        Returns:
            (source1_features, source2_features, source1_names, source2_names)
        """
        idx1 = self.get_feature_indices(feature_names, self.source1_groups)
        idx2 = self.get_feature_indices(feature_names, self.source2_groups)

        # 确保至少有一些特征
        if not idx1:
            idx1 = list(range(len(feature_names) // 2))
        if not idx2:
            idx2 = list(range(len(feature_names) // 2, len(feature_names)))

        X1 = X[:, idx1]
        X2 = X[:, idx2]

        names1 = [feature_names[i] for i in idx1]
        names2 = [feature_names[i] for i in idx2]

        print(f"\n数据源分割:")
        print(f"  Source 1 ({'+'.join(self.source1_groups)}): {X1.shape[1]} 个特征")
        print(f"  Source 2 ({'+'.join(self.source2_groups)}): {X2.shape[1]} 个特征")

        return X1, X2, names1, names2


class DataBalancer:
    """
    数据平衡器

    处理类不平衡问题，支持多种采样策略。
    """

    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        初始化平衡器

        Args:
            method: 采样方法 ('smote', 'adasyn', 'undersample', 'smote_tomek')
            random_state: 随机种子
        """
        self.method = method
        self.random_state = random_state

    def balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡数据集

        Args:
            X: 特征数组
            y: 标签数组
            sampling_strategy: 采样策略

        Returns:
            平衡后的 (X, y)
        """
        print(f"\n=== 数据平衡 (方法: {self.method}) ===")
        print(f"原始分布: {dict(zip(*np.unique(y, return_counts=True)))}")

        if self.method == 'smote':
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == 'adasyn':
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == 'undersample':
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'smote_tomek':
            sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            print(f"未知方法 {self.method}，返回原始数据")
            return X, y

        try:
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            print(f"平衡后分布: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"平衡失败: {e}，返回原始数据")
            return X, y


class DataSplitter:
    """
    数据集划分器

    支持分层采样，确保各类别比例一致。
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        划分数据集为训练/验证/测试集

        Args:
            X: 特征数组
            y: 标签数组
            stratify: 是否分层采样

        Returns:
            包含各数据集的字典
        """
        strat = y if stratify else None

        # 先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat
        )

        # 再从剩余数据分出验证集
        val_ratio = self.val_size / (1 - self.test_size)
        strat_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=strat_temp
        )

        print(f"\n数据集划分:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  验证集: {X_val.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def split_multi_source(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        stratify: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        同时划分多个数据源

        确保各数据源的划分索引一致。
        """
        n_samples = len(y)
        indices = np.arange(n_samples)

        strat = y if stratify else None

        # 划分索引
        idx_temp, idx_test, y_temp, y_test = train_test_split(
            indices, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat
        )

        val_ratio = self.val_size / (1 - self.test_size)
        strat_temp = y_temp if stratify else None

        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=strat_temp
        )

        print(f"\n多源数据集划分:")
        print(f"  训练集: {len(idx_train)} 样本")
        print(f"  验证集: {len(idx_val)} 样本")
        print(f"  测试集: {len(idx_test)} 样本")

        return {
            'X1_train': X1[idx_train], 'X1_val': X1[idx_val], 'X1_test': X1[idx_test],
            'X2_train': X2[idx_train], 'X2_val': X2[idx_val], 'X2_test': X2[idx_test],
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
