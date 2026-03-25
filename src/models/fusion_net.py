"""
Deep learning models for network attack detection.
Supports multiple encoder backbones and fusion strategies.
"""
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AttentionFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        fused = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        return fused, attn_weights


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

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
        batch_size, num_sources, dim = x.shape

        q = self.q_proj(x).view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, num_sources, dim)
        out = self.out_proj(out)
        out = self.norm(out + x)
        fused = out.mean(dim=1)
        return fused, attn_weights.mean(dim=1)


class CrossAttentionFusion(nn.Module):
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
            nn.Linear(dim * 4, dim),
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x1_seq = x1.unsqueeze(1)
        x2_seq = x2.unsqueeze(1)

        x1_attended, attn_1_to_2 = self.cross_attn_1_to_2(x1_seq, x2_seq, x2_seq)
        x2_attended, attn_2_to_1 = self.cross_attn_2_to_1(x2_seq, x1_seq, x1_seq)

        x1_out = self.norm1(x1_seq + x1_attended).squeeze(1)
        x2_out = self.norm2(x2_seq + x2_attended).squeeze(1)

        combined = torch.cat([x1_out, x2_out], dim=-1)
        fused = self.final_norm(self.ffn(combined))

        return fused, {
            "attn_1_to_2": attn_1_to_2.squeeze(1),
            "attn_2_to_1": attn_2_to_1.squeeze(1),
        }


class GatedFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x1, x2], dim=-1)
        gate_values = self.gate(combined)
        transformed = self.transform(combined)

        fused = gate_values * x1 + (1 - gate_values) * x2 + transformed
        gate_mean = gate_values.mean(dim=-1, keepdim=True)
        attention_weights = torch.cat([gate_mean, 1 - gate_mean], dim=-1)
        return fused, attention_weights


class MultiSourceGatedFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = F.softmax(self.score(x).squeeze(-1), dim=1)
        transformed = self.transform(x)
        fused = (transformed * attention_weights.unsqueeze(-1)).sum(dim=1)
        return fused, attention_weights


class BilinearFusion(nn.Module):
    def __init__(self, dim1: int, dim2: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, None]:
        fused = self.bilinear(x1, x2)
        fused = self.dropout(self.norm(fused))
        return fused, None


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        layers = []
        for _ in range(num_layers - 1):
            if use_residual:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
        self.layers = nn.Sequential(*layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.layers(x)
        return self.output_proj(x)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 7]

        n_kernels = len(kernel_sizes)
        channels_per_kernel = hidden_dim // n_kernels
        self.actual_hidden = channels_per_kernel * n_kernels

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, channels_per_kernel, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(channels_per_kernel),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.actual_hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = self.pool(conv_out).squeeze(-1)
            conv_outs.append(conv_out)
        out = torch.cat(conv_outs, dim=-1)
        return self.fc(out)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.chunk_size = 8
        self.seq_len = (input_dim + self.chunk_size - 1) // self.chunk_size
        self.padded_dim = self.seq_len * self.chunk_size

        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if x.size(1) < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = x.view(batch_size, self.seq_len, self.chunk_size)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.chunk_size = 16
        self.seq_len = (input_dim + self.chunk_size - 1) // self.chunk_size
        self.padded_dim = self.seq_len * self.chunk_size

        self.input_proj = nn.Linear(self.chunk_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=self.seq_len + 1, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if x.size(1) < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        x = x.view(batch_size, self.seq_len, self.chunk_size)
        x = self.input_proj(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        cls_out = x[:, 0]
        return self.output_proj(cls_out)


class FusionNet(nn.Module):
    """Fusion network with backward-compatible first two encoder names."""

    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        encoder_type: str = "mlp",
        fusion_type: str = "attention",
        num_layers: int = 2,
        num_heads: int = 4,
        source_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.source_dims = list(source_dims) if source_dims else [traffic_dim, log_dim]
        if len(self.source_dims) < 2:
            raise ValueError("FusionNet requires at least two source dimensions")
        self.num_sources = len(self.source_dims)

        encoder_cls = {
            "mlp": MLPEncoder,
            "cnn": CNNEncoder,
            "lstm": LSTMEncoder,
            "transformer": TransformerEncoder,
        }.get(encoder_type, MLPEncoder)

        encoder_kwargs = {
            "hidden_dim": hidden_dim,
            "output_dim": hidden_dim,
            "dropout": dropout,
        }
        if encoder_type in ["mlp", "lstm", "transformer"]:
            encoder_kwargs["num_layers"] = num_layers
        if encoder_type == "transformer":
            encoder_kwargs["num_heads"] = num_heads

        self.traffic_encoder = encoder_cls(input_dim=self.source_dims[0], **encoder_kwargs)
        self.log_encoder = encoder_cls(input_dim=self.source_dims[1], **encoder_kwargs)
        self.extra_encoders = nn.ModuleList([
            encoder_cls(input_dim=dim, **encoder_kwargs)
            for dim in self.source_dims[2:]
        ])

        if fusion_type == "attention":
            self.fusion = AttentionFusion(hidden_dim, hidden_dim // 2)
            fusion_out_dim = hidden_dim
        elif fusion_type == "multi_head":
            self.fusion = MultiHeadAttentionFusion(hidden_dim, num_heads, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == "cross":
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == "gated":
            self.fusion = GatedFusion(hidden_dim, dropout) if self.num_sources == 2 else MultiSourceGatedFusion(hidden_dim, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(hidden_dim, hidden_dim, hidden_dim, dropout)
            fusion_out_dim = hidden_dim
        elif fusion_type == "concat":
            self.fusion = None
            fusion_out_dim = hidden_dim * self.num_sources
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Linear(fusion_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encoders(self) -> List[nn.Module]:
        return [self.traffic_encoder, self.log_encoder, *self.extra_encoders]

    def _encode_sources(self, source_features: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        if len(source_features) != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} source tensors, received {len(source_features)}")
        return [encoder(features) for encoder, features in zip(self._encoders(), source_features)]

    def _uniform_attention(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.num_sources), 1.0 / self.num_sources, device=device)

    def _normalize_attention(
        self,
        attention: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention is None:
            return self._uniform_attention(batch_size, device)
        if isinstance(attention, dict):
            attn_1_to_2 = attention.get("attn_1_to_2")
            attn_2_to_1 = attention.get("attn_2_to_1")
            if attn_1_to_2 is None or attn_2_to_1 is None:
                return self._uniform_attention(batch_size, device)

            def _sample_mean(attn: torch.Tensor) -> torch.Tensor:
                return attn.reshape(attn.size(0), -1).mean(dim=1, keepdim=True)

            weight_1 = _sample_mean(attn_1_to_2)
            weight_2 = _sample_mean(attn_2_to_1)
            total = (weight_1 + weight_2).clamp_min(1e-8)
            return torch.cat([weight_1 / total, weight_2 / total], dim=1)
        if attention.dim() == 3:
            return attention.mean(dim=1)
        return attention

    def forward(self, *source_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_sources = self._encode_sources(source_features)
        batch_size = source_features[0].size(0)
        device = source_features[0].device

        if self.fusion_type == "concat":
            fused = torch.cat(encoded_sources, dim=-1)
            attention_weights = self._uniform_attention(batch_size, device)
        elif self.fusion_type in ["cross", "bilinear"]:
            if self.num_sources != 2:
                raise ValueError(f"Fusion type '{self.fusion_type}' only supports exactly two sources")
            fused, raw_attention = self.fusion(encoded_sources[0], encoded_sources[1])
            attention_weights = self._normalize_attention(raw_attention, batch_size, device)
        elif self.fusion_type == "gated" and self.num_sources == 2 and isinstance(self.fusion, GatedFusion):
            fused, raw_attention = self.fusion(encoded_sources[0], encoded_sources[1])
            attention_weights = self._normalize_attention(raw_attention, batch_size, device)
        else:
            combined = torch.stack(encoded_sources, dim=1)
            fused, raw_attention = self.fusion(combined)
            attention_weights = self._normalize_attention(raw_attention, batch_size, device)

        logits = self.classifier(fused)
        return logits, attention_weights

    def get_attention_weights(self, *source_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(*source_features)
        return {"fusion_attention": attn}


class AgenticDecisionController:
    """Rule-guided inference controller for threat-intel assisted decisions."""

    def __init__(
        self,
        num_classes: int,
        uncertainty_threshold: float = 0.55,
        intel_confidence_threshold: float = 0.70,
        suspicious_class_indices: Optional[List[int]] = None,
        benign_class_idx: int = 0,
        consensus_boost: float = 0.15,
    ):
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.intel_confidence_threshold = intel_confidence_threshold
        self.suspicious_class_indices = set(suspicious_class_indices or list(range(1, num_classes)))
        self.benign_class_idx = benign_class_idx
        self.consensus_boost = consensus_boost

    def apply(
        self,
        network_logits: torch.Tensor,
        intel_logits: torch.Tensor,
        fused_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        network_probs = torch.softmax(network_logits, dim=-1)
        intel_probs = torch.softmax(intel_logits, dim=-1)

        net_conf, net_pred = network_probs.max(dim=-1)
        intel_conf, intel_pred = intel_probs.max(dim=-1)

        adjusted_logits = fused_logits.clone()
        actions = []

        for idx in range(fused_logits.size(0)):
            action = "no_override"
            if net_conf[idx].item() < self.uncertainty_threshold and intel_conf[idx].item() >= self.intel_confidence_threshold:
                adjusted_logits[idx] = 0.30 * network_logits[idx] + 0.70 * intel_logits[idx]
                action = "intel_override_uncertain_network"
            elif (
                net_pred[idx].item() == self.benign_class_idx
                and intel_pred[idx].item() in self.suspicious_class_indices
                and intel_conf[idx].item() >= self.intel_confidence_threshold
            ):
                adjusted_logits[idx] = 0.20 * network_logits[idx] + 0.80 * intel_logits[idx]
                action = "intel_override_benign_disagreement"
            elif net_pred[idx].item() == intel_pred[idx].item() and intel_conf[idx].item() >= self.intel_confidence_threshold:
                adjusted_logits[idx, net_pred[idx]] = adjusted_logits[idx, net_pred[idx]] + self.consensus_boost
                action = "consensus_boost"
            actions.append(action)

        return adjusted_logits, actions


class DecisionLevelFusionNet(nn.Module):
    """Flow/log feature fusion with threat-intel decision-level fusion."""

    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        encoder_type: str = "mlp",
        fusion_type: str = "attention",
        num_layers: int = 2,
        num_heads: int = 4,
        source_dims: Optional[List[int]] = None,
        threat_intel_dim: Optional[int] = None,
        decision_hidden_dim: Optional[int] = None,
        threat_encoder_type: str = "mlp",
        agentic_mode: Optional[Dict] = None,
    ):
        super().__init__()

        self.source_dims = list(source_dims) if source_dims else [traffic_dim, log_dim]
        if len(self.source_dims) < 2:
            raise ValueError("DecisionLevelFusionNet requires at least flow and log source dims")
        self.feature_source_dims = self.source_dims[:2]
        self.threat_intel_dim = threat_intel_dim or (self.source_dims[2] if len(self.source_dims) > 2 else None)
        if self.threat_intel_dim is None:
            raise ValueError("DecisionLevelFusionNet requires a third source dimension for threat-intel input")

        self.feature_fusion = FusionNet(
            traffic_dim=self.feature_source_dims[0],
            log_dim=self.feature_source_dims[1],
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            encoder_type=encoder_type,
            fusion_type=fusion_type,
            num_layers=num_layers,
            num_heads=num_heads,
            source_dims=self.feature_source_dims,
        )

        encoder_cls = {
            "mlp": MLPEncoder,
            "cnn": CNNEncoder,
            "lstm": LSTMEncoder,
            "transformer": TransformerEncoder,
        }.get(threat_encoder_type, MLPEncoder)
        encoder_kwargs = {
            "hidden_dim": hidden_dim,
            "output_dim": hidden_dim,
            "dropout": dropout,
        }
        if threat_encoder_type in ["mlp", "lstm", "transformer"]:
            encoder_kwargs["num_layers"] = num_layers
        if threat_encoder_type == "transformer":
            encoder_kwargs["num_heads"] = num_heads

        self.threat_intel_encoder = encoder_cls(input_dim=self.threat_intel_dim, **encoder_kwargs)
        self.threat_intel_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        gate_hidden_dim = decision_hidden_dim or hidden_dim
        self.decision_gate = nn.Sequential(
            nn.Linear(num_classes * 2 + hidden_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.agentic_config = agentic_mode or {}
        self.agentic_enabled = self.agentic_config.get("enabled", False)
        suspicious_indices = self.agentic_config.get("suspicious_class_indices")
        self.agentic_controller = AgenticDecisionController(
            num_classes=num_classes,
            uncertainty_threshold=self.agentic_config.get("uncertainty_threshold", 0.55),
            intel_confidence_threshold=self.agentic_config.get("intel_confidence_threshold", 0.70),
            suspicious_class_indices=suspicious_indices,
            benign_class_idx=self.agentic_config.get("benign_class_idx", 0),
            consensus_boost=self.agentic_config.get("consensus_boost", 0.15),
        )
        self.last_agentic_actions: List[str] = []

    def forward(self, *source_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(source_features) < 3:
            raise ValueError("DecisionLevelFusionNet expects flow, log, and threat-intel features")

        flow_features, log_features, threat_intel_features = source_features[:3]
        network_logits, network_attention = self.feature_fusion(flow_features, log_features)

        intel_repr = self.threat_intel_encoder(threat_intel_features)
        intel_logits = self.threat_intel_classifier(intel_repr)

        gate_input = torch.cat([network_logits, intel_logits, intel_repr], dim=-1)
        network_weight = self.decision_gate(gate_input)
        fused_logits = network_weight * network_logits + (1.0 - network_weight) * intel_logits

        if self.agentic_enabled and not self.training:
            fused_logits, self.last_agentic_actions = self.agentic_controller.apply(
                network_logits,
                intel_logits,
                fused_logits,
            )
        else:
            self.last_agentic_actions = []

        combined_attention = torch.cat(
            [network_attention * network_weight, 1.0 - network_weight],
            dim=-1,
        )
        return fused_logits, combined_attention


class SingleSourceNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        encoder_type: str = "mlp",
        num_layers: int = 3,
    ):
        super().__init__()

        encoder_cls = {
            "mlp": MLPEncoder,
            "cnn": CNNEncoder,
            "lstm": LSTMEncoder,
            "transformer": TransformerEncoder,
        }.get(encoder_type, MLPEncoder)

        encoder_kwargs = {
            "hidden_dim": hidden_dim,
            "output_dim": hidden_dim,
            "dropout": dropout,
        }
        if encoder_type in ["mlp", "lstm", "transformer"]:
            encoder_kwargs["num_layers"] = num_layers

        self.encoder = encoder_cls(input_dim=input_dim, **encoder_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.classifier(encoded)


class EnsembleFusionNet(nn.Module):
    def __init__(
        self,
        traffic_dim: int,
        log_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion_types: Optional[List[str]] = None,
        source_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        fusion_types = fusion_types or ["attention", "gated", "cross"]
        self.models = nn.ModuleList([
            FusionNet(
                traffic_dim=traffic_dim,
                log_dim=log_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
                fusion_type=ft,
                source_dims=source_dims,
            )
            for ft in fusion_types
        ])
        self.ensemble_weights = nn.Parameter(torch.ones(len(fusion_types)) / len(fusion_types))

    def forward(self, *source_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        attentions = []
        for model in self.models:
            out, attn = model(*source_features)
            outputs.append(out)
            attentions.append(attn)

        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = sum(w * o for w, o in zip(weights, outputs))
        mean_attention = torch.stack(attentions).mean(dim=0)
        return ensemble_output, mean_attention


def create_model(
    model_type: str,
    traffic_dim: int,
    log_dim: int,
    num_classes: int,
    config: Dict = None,
) -> nn.Module:
    config = config or {}
    source_dims = config.get("source_dims") or [traffic_dim, log_dim]
    decision_config = config.get("decision_fusion", {})
    agentic_config = config.get("agentic_mode", {})

    if model_type == "fusion_net":
        return FusionNet(
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            num_classes=num_classes,
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
            encoder_type=config.get("encoder_type", "mlp"),
            fusion_type=config.get("fusion_type", "attention"),
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 4),
            source_dims=source_dims,
        )
    if model_type == "decision_fusion_net":
        return DecisionLevelFusionNet(
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            num_classes=num_classes,
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
            encoder_type=config.get("encoder_type", "mlp"),
            fusion_type=config.get("fusion_type", "attention"),
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 4),
            source_dims=source_dims,
            threat_intel_dim=decision_config.get("threat_intel_dim"),
            decision_hidden_dim=decision_config.get("hidden_dim"),
            threat_encoder_type=decision_config.get("threat_encoder_type", "mlp"),
            agentic_mode=agentic_config,
        )
    if model_type == "single_source":
        return SingleSourceNet(
            input_dim=sum(source_dims),
            num_classes=num_classes,
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
            encoder_type=config.get("encoder_type", "mlp"),
            num_layers=config.get("num_layers", 3),
        )
    if model_type == "ensemble":
        return EnsembleFusionNet(
            traffic_dim=traffic_dim,
            log_dim=log_dim,
            num_classes=num_classes,
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
            fusion_types=config.get("fusion_types", ["attention", "gated", "cross"]),
            source_dims=source_dims,
        )
    raise ValueError(f"Unknown model type: {model_type}")
