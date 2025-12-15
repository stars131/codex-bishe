"""
Deep learning models for network attack detection.
Supports CNN, LSTM, Transformer, and various Fusion architectures.

Author: Network Attack Detection Project
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Union


# ============================================
# 基础模块
# ============================================

class ResidualBlock(nn.Module):
    """残差连接块"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))


class PositionalEncoding(nn.Module):
    """位置编码（用于Transformer）"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================
# 注意力融合模块
# ============================================

class AttentionFusion(nn.Module):
    """基础注意力融合模块"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_sources, dim)
        Returns:
            fused: (batch, dim)
            attention_weights: (batch, num_sources)
        """
        attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        fused = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        return fused, attn_weights


class MultiHeadAttentionFusion(nn.Module):
    """多头注意力融合模块"""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_sources, dim)
        Returns:
            fused: (batch, dim)
            attention_weights: (batch, num_heads, num_sources, num_sources)
        """
        batch_size, num_sources, dim = x.shape

        # 投影
        q = self.q_proj(x).view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 输出
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, num_sources, dim)
        out = self.out_proj(out)
        out = self.norm(out + x)  # 残差连接

        # 融合所有源
        fused = out.mean(dim=1)

        return fused, attn_weights.mean(dim=1)  # 返回平均注意力权重


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块 - 两个数据源之间的双向注意力"""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_1_to_2 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_2_to_1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x1: (batch, dim) - 源1特征
            x2: (batch, dim) - 源2特征
        Returns:
            fused: (batch, dim)
            attention_info: 包含注意力权重的字典
        """
        # 扩展维度用于attention
        x1_seq = x1.unsqueeze(1)  # (batch, 1, dim)
        x2_seq = x2.unsqueeze(1)  # (batch, 1, dim)

        # 交叉注意力
        x1_attended, attn_1_to_2 = self.cross_attn_1_to_2(x1_seq, x2_seq, x2_seq)
        x2_attended, attn_2_to_1 = self.cross_attn_2_to_1(x2_seq, x1_seq, x1_seq)

        x1_out = self.norm1(x1_seq + x1_attended).squeeze(1)
        x2_out = self.norm2(x2_seq + x2_attended).squeeze(1)

        # 融合
        combined = torch.cat([x1_out, x2_out], dim=-1)
        fused = self.final_norm(self.ffn(combined))

        attention_info = {
            'attn_1_to_2': attn_1_to_2.squeeze(1),
            'attn_2_to_1': attn_2_to_1.squeeze(1)
        }

        return fused, attention_info


class GatedFusion(nn.Module):
    """门控融合模块"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: (batch, dim) - 源1特征
            x2: (batch, dim) - 源2特征
        Returns:
            fused: (batch, dim)
            gate_values: (batch, dim) - 门控值
        """
        combined = torch.cat([x1, x2], dim=-1)
        gate_values = self.gate(combined)
        transformed = self.transform(combined)

        fused = gate_values * x1 + (1 - gate_values) * x2 + transformed

        return fused, gate_values.mean(dim=-1, keepdim=True)


class BilinearFusion(nn.Module):
    """双线性融合模块"""

    def __init__(self, dim1: int, dim2: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x1: (batch, dim1)
            x2: (batch, dim2)
        Returns:
            fused: (batch, out_dim)
            None: 占位符，保持接口一致
        """
        fused = self.bilinear(x1, x2)
        fused = self.dropout(self.norm(fused))
        return fused, None


# ============================================
# 特征编码器
# ============================================

class MLPEncoder(nn.Module):
    """多层感知机编码器"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 中间层
        layers = []
        for _ in range(num_layers - 1):
            if use_residual:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
        self.layers = nn.Sequential(*layers)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.layers(x)
        return self.output_proj(x)


class CNNEncoder(nn.Module):
    """1D卷积编码器 - 用于捕获局部模式"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.3
    ):
        super().__init__()

        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim // len(kernel_sizes), kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                nn.GELU()
            )
            for k in kernel_sizes
        ])

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)

        # 多尺度卷积
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(x)  # (batch, hidden//n, features)
            conv_out = self.pool(conv_out).squeeze(-1)  # (batch, hidden//n)
            conv_outs.append(conv_out)

        # 拼接
        out = torch.cat(conv_outs, dim=-1)  # (batch, hidden)
        return self.fc(out)


class LSTMEncoder(nn.Module):
    """LSTM编码器 - 将特征视为序列"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        # 将特征分组为序列
        self.chunk_size = 8  # 每个时间步的特征数
        self.seq_len = (input_dim + self.chunk_size - 1) // self.chunk_size
        self.padded_dim = self.seq_len * self.chunk_size

        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 填充到可整除的长度
        if x.size(1) < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        # 重塑为序列
        x = x.view(batch_size, self.seq_len, self.chunk_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后时间步的输出
        out = lstm_out[:, -1, :]

        return self.fc(out)


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        # 将特征分组为序列
        self.chunk_size = 16
        self.seq_len = (input_dim + self.chunk_size - 1) // self.chunk_size
        self.padded_dim = self.seq_len * self.chunk_size

        # 输入投影
        self.input_proj = nn.Linear(self.chunk_size, hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=self.seq_len + 1, dropout=dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 填充
        if x.size(1) < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        # 重塑为序列
        x = x.view(batch_size, self.seq_len, self.chunk_size)

        # 投影
        x = self.input_proj(x)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # 使用CLS token的输出
        cls_out = x[:, 0]

        return self.output_proj(cls_out)


# ============================================
# 主要融合网络
# ============================================

class FusionNet(nn.Module):
    """
    多源数据融合网络

    支持多种编码器和融合方法的灵活组合。
    """

    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        encoder_type: str = 'mlp',
        fusion_type: str = 'attention',
        num_layers: int = 2,
        num_heads: int = 4
    ):
        """
        Args:
            traffic_dim: 流量特征维度
            log_dim: 日志特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            dropout: Dropout比率
            encoder_type: 编码器类型 ('mlp', 'cnn', 'lstm', 'transformer')
            fusion_type: 融合类型 ('attention', 'multi_head', 'cross', 'gated', 'bilinear', 'concat')
            num_layers: 编码器层数
            num_heads: 注意力头数
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim

        # 选择编码器
        EncoderClass = {
            'mlp': MLPEncoder,
            'cnn': CNNEncoder,
            'lstm': LSTMEncoder,
            'transformer': TransformerEncoder
        }.get(encoder_type, MLPEncoder)

        encoder_kwargs = {
            'hidden_dim': hidden_dim,
            'output_dim': hidden_dim,
            'dropout': dropout
        }
        if encoder_type in ['mlp', 'lstm', 'transformer']:
            encoder_kwargs['num_layers'] = num_layers
        if encoder_type == 'transformer':
            encoder_kwargs['num_heads'] = num_heads

        self.traffic_encoder = EncoderClass(input_dim=traffic_dim, **encoder_kwargs)
        self.log_encoder = EncoderClass(input_dim=log_dim, **encoder_kwargs)

        # 选择融合模块
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(hidden_dim, hidden_dim // 2)
            fusion_out_dim = hidden_dim
        elif fusion_type == 'multi_head':
            self.fusion = MultiHeadAttentionFusion(hidden_dim, num_heads, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == 'cross':
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(hidden_dim, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == 'bilinear':
            self.fusion = BilinearFusion(hidden_dim, hidden_dim, hidden_dim, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == 'concat':
            self.fusion = None
            fusion_out_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        traffic_features: torch.Tensor,
        log_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            traffic_features: 流量特征 (batch, traffic_dim)
            log_features: 日志特征 (batch, log_dim)

        Returns:
            logits: 分类logits (batch, num_classes)
            attention_weights: 注意力权重（形状取决于融合类型）
        """
        # 编码
        traffic_encoded = self.traffic_encoder(traffic_features)
        log_encoded = self.log_encoder(log_features)

        # 融合
        if self.fusion_type == 'concat':
            fused = torch.cat([traffic_encoded, log_encoded], dim=-1)
            attention_weights = torch.tensor([0.5, 0.5], device=traffic_features.device)
            attention_weights = attention_weights.unsqueeze(0).expand(traffic_features.size(0), -1)
        elif self.fusion_type in ['cross', 'gated', 'bilinear']:
            fused, attention_weights = self.fusion(traffic_encoded, log_encoded)
            if attention_weights is None:
                attention_weights = torch.tensor([0.5, 0.5], device=traffic_features.device)
                attention_weights = attention_weights.unsqueeze(0).expand(traffic_features.size(0), -1)
        else:
            # attention, multi_head
            combined = torch.stack([traffic_encoded, log_encoded], dim=1)
            fused, attention_weights = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        return logits, attention_weights

    def get_attention_weights(
        self,
        traffic_features: torch.Tensor,
        log_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """获取详细的注意力权重信息"""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(traffic_features, log_features)
        return {'fusion_attention': attn}


class SingleSourceNet(nn.Module):
    """单源网络 - 用于消融实验对比"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        encoder_type: str = 'mlp',
        num_layers: int = 3
    ):
        super().__init__()

        # 选择编码器
        EncoderClass = {
            'mlp': MLPEncoder,
            'cnn': CNNEncoder,
            'lstm': LSTMEncoder,
            'transformer': TransformerEncoder
        }.get(encoder_type, MLPEncoder)

        encoder_kwargs = {
            'hidden_dim': hidden_dim,
            'output_dim': hidden_dim,
            'dropout': dropout
        }
        if encoder_type in ['mlp', 'lstm', 'transformer']:
            encoder_kwargs['num_layers'] = num_layers

        self.encoder = EncoderClass(input_dim=input_dim, **encoder_kwargs)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.classifier(encoded)


class EnsembleFusionNet(nn.Module):
    """集成融合网络 - 组合多种融合方法"""

    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion_types: List[str] = ['attention', 'gated', 'cross']
    ):
        super().__init__()

        self.models = nn.ModuleList([
            FusionNet(
                traffic_dim=traffic_dim,
                log_dim=log_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
                fusion_type=ft
            )
            for ft in fusion_types
        ])

        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(len(fusion_types)) / len(fusion_types))

    def forward(
        self,
        traffic_features: torch.Tensor,
        log_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        attentions = []

        for model in self.models:
            out, attn = model(traffic_features, log_features)
            outputs.append(out)
            attentions.append(attn)

        # 加权集成
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = sum(w * o for w, o in zip(weights, outputs))

        # 平均注意力
        mean_attention = torch.stack(attentions).mean(dim=0)

        return ensemble_output, mean_attention


# ============================================
# 工厂函数
# ============================================

def create_model(
    model_type: str,
    traffic_dim: int,
    log_dim: int,
    num_classes: int,
    config: Dict = None
) -> nn.Module:
    """
    模型工厂函数

    Args:
        model_type: 模型类型
        traffic_dim: 流量特征维度
        log_dim: 日志特征维度
        num_classes: 类别数
        config: 额外配置

    Returns:
        模型实例
    """
    config = config or {}

    if model_type == 'fusion_net':
        return FusionNet(
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            num_classes=num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.3),
            encoder_type=config.get('encoder_type', 'mlp'),
            fusion_type=config.get('fusion_type', 'attention'),
            num_layers=config.get('num_layers', 2),
            num_heads=config.get('num_heads', 4)
        )
    elif model_type == 'single_source':
        return SingleSourceNet(
            input_dim=traffic_dim + log_dim,
            num_classes=num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.3),
            encoder_type=config.get('encoder_type', 'mlp'),
            num_layers=config.get('num_layers', 3)
        )
    elif model_type == 'ensemble':
        return EnsembleFusionNet(
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            num_classes=num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.3),
            fusion_types=config.get('fusion_types', ['attention', 'gated', 'cross'])
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
