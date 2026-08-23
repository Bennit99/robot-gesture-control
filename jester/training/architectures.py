"""Pluggable model architectures for Jester gesture classification.

All models follow the same interface:
    - Constructor: ``(input_dim, num_classes, seq_len, **kwargs)``
    - Forward:     ``(x: Tensor[B, T, D])  logits: Tensor[B, C]``

Register new architectures by adding them to ``ARCHITECTURE_REGISTRY``
at the bottom of this file.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ==============================================================================
# 1. LSTM
# ==============================================================================
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM  mean-pool  classifier head.

    A baseline for variable-length temporal sequences.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 40,
        hidden_dim: int = 224,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.lstm(x)  # (B, T, H)
        pooled = out.mean(dim=1)  # (B, H)
        return self.head(pooled)  # (B, C)


# ==============================================================================
# 2. GRU
# ==============================================================================
class GRUClassifier(nn.Module):
    """Bidirectional GRU - lighter / faster variant of the LSTM baseline."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)


# ==============================================================================
# 3. Transformer Encoder
# ==============================================================================
class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    """Transformer Encoder with CLS-token pooling.

    Projects landmarks to ``d_model``, prepends a learnable [CLS] token,
    runs through a standard Transformer encoder, and classifies from
    the [CLS] output.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 40,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **_kwargs,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(
            d_model, max_len=seq_len + 1, dropout=dropout
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.input_proj(x)  # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = x[:, 0]  # (B, d_model)
        return self.head(cls_out)  # (B, C)


# ==============================================================================
# 4. 1D-CNN (Temporal Convolution)
# ==============================================================================
class Conv1DClassifier(nn.Module):
    """Stacked 1D convolutions over the temporal axis  global avg pool  head.

    Good at capturing short-range temporal patterns (like quick swipes).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 40,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        **_kwargs,
    ):
        super().__init__()
        channels = channels or [312, 624, 1248]
        layers: list[nn.Module] = []
        in_ch = input_dim

        dilations = [1, 2, 4]

        for out_ch, d in zip(channels, dilations):
            pad = (kernel_size - 1) * d // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  Conv1d expects (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)  # (B, C_last, T)
        x = x.mean(dim=2)  # global avg pool  (B, C_last)
        return self.head(x)  # (B, num_classes)


# ==============================================================================
# 5. ST-GCN (MediaPipe hand, lightweight)
# ==============================================================================
def build_mediapipe_hand_adjacency(
    num_nodes: int = 21,
    add_self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Build a 21x21 MediaPipe hand adjacency matrix.

    Node layout:
    - 0: wrist
    - 1-4: thumb
    - 5-8: index
    - 9-12: middle
    - 13-16: ring
    - 17-20: pinky

    Includes finger chains + palm links and optional self-loops.
    If *normalize* is True, returns symmetric normalization:
    A_norm = D^(-0.5) A D^(-0.5)
    """
    if num_nodes != 21:
        raise ValueError("MediaPipe hand graph expects exactly 21 nodes.")

    # Undirected edges
    edges: list[tuple[int, int]] = [
        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        # Middle
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        # Ring
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        # Palm cross-links
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    if add_self_loops:
        adjacency += torch.eye(num_nodes, dtype=torch.float32)

    if not normalize:
        return adjacency

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = degree.clamp(min=1e-8).pow(-0.5)
    adjacency_norm = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
    return adjacency_norm


class STGCNBlock(nn.Module):
    """Spatio-temporal graph conv block with residual connection.

    Input:  x in shape (B, C, T, V)
    Output: y in shape (B, C_out, T_out, V)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        padding_t = (temporal_kernel_size - 1) // 2

        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(out_channels)

        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1),
            stride=(stride, 1),
            padding=(padding_t, 0),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        # Spatial graph conv: W * X * A_norm
        x = self.spatial_conv(x)
        x = torch.einsum("bctv,vw->bctw", x, adjacency_norm)
        x = self.spatial_bn(x)
        x = self.relu(x)

        # Temporal conv over time axis
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.dropout(x)

        x = x + residual
        x = self.relu(x)
        return x


class MediaPipeSTGCN(nn.Module):
    """Lightweight ST-GCN for MediaPipe hand landmarks.

    Supports input in either format:
    - (B, T, 63): flattened xyz landmarks
    - (B, 3, T, 21): channel-first ST-GCN layout
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 40,
        in_channels: int = 3,
        num_nodes: int = 21,
        channels: tuple[int, int, int] = (64, 64, 128),
        temporal_kernel_size: int = 9,
        dropout: float = 0.3,
        **_kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        expected_input_dim = in_channels * num_nodes
        if input_dim != expected_input_dim:
            raise ValueError(
                f"MediaPipeSTGCN expects input_dim={expected_input_dim} "
                f"for 1 hand ({in_channels}x{num_nodes}), got {input_dim}."
            )

        adjacency_norm = build_mediapipe_hand_adjacency(
            num_nodes=num_nodes,
            add_self_loops=True,
            normalize=True,
        )
        self.register_buffer("adjacency_norm", adjacency_norm)

        # Input normalization over (C*V, T), as in ST-GCN style data norm.
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        c1, c2, c3 = channels
        self.block1 = STGCNBlock(
            in_channels,
            c1,
            temporal_kernel_size=temporal_kernel_size,
            stride=1,
            dropout=dropout,
        )
        self.block2 = STGCNBlock(
            c1,
            c2,
            temporal_kernel_size=temporal_kernel_size,
            stride=1,
            dropout=dropout,
        )
        self.block3 = STGCNBlock(
            c2,
            c3,
            temporal_kernel_size=temporal_kernel_size,
            stride=2,
            dropout=dropout,
        )

        self.classifier = nn.Linear(c3, num_classes)

    def _to_stgcn_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to (B, C, T, V)."""
        if x.dim() == 3:
            # (B, T, C*V) -> (B, C, T, V)
            batch, time_steps, feat_dim = x.shape
            expected = self.in_channels * self.num_nodes
            if feat_dim != expected:
                raise ValueError(
                    f"Expected flattened feature dim {expected}, got {feat_dim}."
                )
            x = x.view(batch, time_steps, self.in_channels, self.num_nodes)
            x = x.permute(0, 2, 1, 3).contiguous()
            return x

        if x.dim() == 4:
            # Already (B, C, T, V)
            if x.size(1) != self.in_channels or x.size(3) != self.num_nodes:
                raise ValueError(
                    "4D input must have shape (B, C, T, V) with "
                    f"C={self.in_channels}, V={self.num_nodes}."
                )
            return x

        raise ValueError("Input must be 3D (B,T,D) or 4D (B,C,T,V).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_stgcn_layout(x)  # (B, C, T, V)

        # Data BN in (B, C*V, T) layout
        batch, channels, time_steps, nodes = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, nodes * channels, time_steps)
        x = self.data_bn(x)
        x = x.view(batch, nodes, channels, time_steps).permute(0, 2, 3, 1).contiguous()

        x = self.block1(x, self.adjacency_norm)
        x = self.block2(x, self.adjacency_norm)
        x = self.block3(x, self.adjacency_norm)

        # Global average pooling over time and vertices
        x = x.mean(dim=(2, 3))
        logits = self.classifier(x)
        return logits


# ==============================================================================
# Registry
# ==============================================================================
ARCHITECTURE_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "transformer": TransformerClassifier,
    "conv1d": Conv1DClassifier,
    "stgcn": MediaPipeSTGCN,
}


def build_model(
    arch: str,
    input_dim: int,
    num_classes: int,
    seq_len: int,
    **kwargs,
) -> nn.Module:
    """Instantiate a model by name from the registry.

    Parameters
    ----------
    arch        : architecture key (``lstm``, ``gru``, ``transformer``,
                  ``conv1d``, ``stgcn``)
    input_dim   : per-frame feature dimension (e.g. 63 for 1 hand)
    num_classes : number of gesture classes (27 for Jester)
    seq_len     : fixed sequence length (frames)
    **kwargs    : forwarded to the model constructor

    Raises
    ------
    ValueError  if *arch* is not in the registry.
    """
    if arch not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY))
        raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")
    cls = ARCHITECTURE_REGISTRY[arch]
    return cls(input_dim=input_dim, num_classes=num_classes, seq_len=seq_len, **kwargs)
