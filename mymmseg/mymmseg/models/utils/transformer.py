"""Transformer utilities: FFN, DropPath.

Self-contained replacements for mmcv's FFN and build_dropout.
"""

import torch
import torch.nn as nn
from ...utils.layers import build_act_layer


# ── DropPath (Stochastic Depth) ───────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Reference: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # (B, 1, 1, ...) broadcast over all dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


def build_dropout(dropout_cfg: dict) -> nn.Module:
    """Build dropout layer từ config dict.

    Args:
        dropout_cfg: e.g. dict(type='DropPath', drop_prob=0.1)
                          dict(type='Dropout', p=0.1)

    Returns:
        nn.Module
    """
    cfg = dropout_cfg.copy()
    dropout_type = cfg.pop('type', 'Dropout')

    if dropout_type == 'DropPath':
        return DropPath(drop_prob=cfg.get('drop_prob', 0.0))
    elif dropout_type == 'Dropout':
        return nn.Dropout(p=cfg.get('p', 0.5))
    elif dropout_type == 'Dropout2d':
        return nn.Dropout2d(p=cfg.get('p', 0.5))
    else:
        raise ValueError(f"Unknown dropout type: '{dropout_type}'")


# ── FFN ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """Feed-Forward Network dùng trong Transformer.

    Architecture:
        Linear(embed_dims → feedforward_channels)
        → act
        → Dropout
        → Linear(feedforward_channels → embed_dims)
        → Dropout
        → (+ identity nếu add_identity=True)

    Args:
        embed_dims:           Input/output channels. Default: 256.
        feedforward_channels: Hidden channels. Default: 1024.
        num_fcs:              Số Linear layers. Default: 2.
        act_cfg:              Activation config. Default: GELU.
        ffn_drop:             Dropout sau activation. Default: 0.
        dropout_layer:        DropPath config. Default: None.
        add_identity:         Add residual connection. Default: True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act_cfg: dict = dict(type='GELU'),
        ffn_drop: float = 0.0,
        dropout_layer: dict = None,
        add_identity: bool = True,
        init_cfg=None,
    ):
        super().__init__()
        assert num_fcs >= 2, 'num_fcs phải >= 2'

        self.embed_dims         = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs            = num_fcs
        self.add_identity       = add_identity

        layers = []
        in_channels = embed_dims
        for i in range(num_fcs - 1):
            layers += [
                nn.Linear(in_channels, feedforward_channels),
                build_act_layer(act_cfg),
                nn.Dropout(ffn_drop),
            ]
            in_channels = feedforward_channels
        layers += [
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        ]
        self.layers = nn.Sequential(*layers)

        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        identity: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.layers(x)
        out = self.dropout_layer(out)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out