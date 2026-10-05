"""Swin Transformer backbone — API-compatible with mmseg's SwinTransformer.

Self-contained: no mmcv, no mmengine deps.
Reference: https://arxiv.org/abs/2103.14030
"""

import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from ...utils.registry import BACKBONES
from ...utils.base_module import BaseModule, ModuleList
from ...utils.layers import build_norm_layer
from ..utils.embed import PatchEmbed, PatchMerging, _build_norm_1d
from ..utils.transformer import FFN, DropPath, build_dropout


# ── Window MSA ────────────────────────────────────────────────────────────────

class WindowMSA(BaseModule):
    """Window-based Multi-head Self-Attention with relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.embed_dims  = embed_dims
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = embed_dims // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        # Relative position bias table: (2*Wh-1) * (2*Ww-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position index
        Wh, Ww = window_size
        rel_index = self._double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index + rel_index.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv       = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj      = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q   = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        ).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x    = self.proj_drop(self.proj(x))
        return x

    @staticmethod
    def _double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# ── Shift-Window MSA ──────────────────────────────────────────────────────────

class ShiftWindowMSA(BaseModule):
    """Shifted Window Multi-head Self-Attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        dropout_layer: dict = dict(type='DropPath', drop_prob=0.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.window_size = window_size
        self.shift_size  = shift_size
        assert 0 <= shift_size < window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=(window_size, window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = build_dropout(dropout_layer)

    def forward(self, query: torch.Tensor, hw_shape: Tuple[int, int]):
        B, L, C = query.shape
        H, W    = hw_shape
        assert L == H * W
        query = query.view(B, H, W, C)

        # Pad to multiple of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            shifted = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
            attn_mask = self._compute_mask(H_pad, W_pad, query.device)
        else:
            shifted    = query
            attn_mask  = None

        # Partition into windows
        q_windows = self._window_partition(shifted)           # (nW*B, ws, ws, C)
        q_windows = q_windows.view(-1, self.window_size ** 2, C)

        # Attention
        attn_out = self.w_msa(q_windows, mask=attn_mask)     # (nW*B, ws*ws, C)

        # Reverse windows
        attn_out  = attn_out.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_out, H_pad, W_pad)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        return self.drop(x)

    def _compute_mask(self, H, W, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        ws, ss   = self.window_size, self.shift_size
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = self._window_partition(img_mask)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask    = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def _window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        x  = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

    def _window_reverse(self, windows, H, W):
        ws = self.window_size
        B  = int(windows.shape[0] / (H * W / ws / ws))
        x  = windows.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Swin Block ────────────────────────────────────────────────────────────────

class SwinBlock(BaseModule):
    """One Swin Transformer block: W-MSA (or SW-MSA) + FFN."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN'),
        with_cp: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.with_cp = with_cp

        self.norm1 = _build_norm_1d(norm_cfg, embed_dims)
        self.attn  = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        )

        self.norm2 = _build_norm_1d(norm_cfg, embed_dims)
        self.ffn   = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            add_identity=True,
        )

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]):
        def _inner(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner, x)
        return _inner(x)


# ── Swin Block Sequence (one stage) ──────────────────────────────────────────

class SwinBlockSequence(BaseModule):
    """One Swin stage = depth × SwinBlock + optional PatchMerging."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: Union[float, List[float]] = 0.0,
        downsample=None,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN'),
        with_cp: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        if isinstance(drop_path_rate, (int, float)):
            drop_path_rates = [deepcopy(drop_path_rate)] * depth
        else:
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth

        self.blocks = ModuleList([
            SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=(i % 2 != 0),   # alternate W-MSA / SW-MSA
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
            )
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw = self.downsample(x, hw_shape)
            return x_down, down_hw, x, hw_shape
        return x, hw_shape, x, hw_shape


# ── SwinTransformer ───────────────────────────────────────────────────────────

@BACKBONES.register_module()
class SwinTransformer(BaseModule):
    """Swin Transformer backbone — API-compatible with mmseg.

    Args:
        pretrain_img_size: Input image size used during pretraining. Default: 224.
        in_channels:       Input channels. Default: 3.
        embed_dims:        Patch embedding dims. Default: 96.
        patch_size:        Patch size. Default: 4.
        window_size:       Attention window size. Default: 7.
        mlp_ratio:         FFN hidden / embed ratio. Default: 4.
        depths:            Blocks per stage. Default: (2,2,6,2).
        num_heads:         Attention heads per stage. Default: (3,6,12,24).
        strides:           Stride per stage (first = patch stride). Default: (4,2,2,2).
        out_indices:       Which stage outputs to return. Default: (0,1,2,3).
        qkv_bias:          QKV bias. Default: True.
        patch_norm:        Norm after patch embed/merge. Default: True.
        drop_rate:         Dropout rate. Default: 0.
        attn_drop_rate:    Attention dropout. Default: 0.
        drop_path_rate:    Stochastic depth rate. Default: 0.1.
        use_abs_pos_embed: Add absolute position embedding. Default: False.
        act_cfg:           Activation config. Default: GELU.
        norm_cfg:          Norm config. Default: LN.
        with_cp:           Gradient checkpointing. Default: False.
        frozen_stages:     Freeze first N stages. Default: -1.
        pretrained:        Path to pretrained weights. Default: None.
        init_cfg:          Init config. Default: None.
    """

    def __init__(
        self,
        pretrain_img_size: int = 224,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        strides: Sequence[int] = (4, 2, 2, 2),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_abs_pos_embed: bool = False,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN'),
        with_cp: bool = False,
        frozen_stages: int = -1,
        pretrained: Optional[str] = None,
        init_cfg=None,
    ):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot both be set'
        if isinstance(pretrained, str):
            warnings.warn('pretrained is deprecated — use init_cfg instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        super().__init__(init_cfg)

        self.frozen_stages     = frozen_stages
        self.out_indices       = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        num_layers             = len(depths)

        assert strides[0] == patch_size, 'strides[0] must equal patch_size'

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
        )

        # Absolute position embedding
        if use_abs_pos_embed:
            patch_size_ = pretrain_img_size // patch_size
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, patch_size_ * patch_size_, embed_dims)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build stages
        self.stages    = ModuleList()
        in_ch          = embed_dims
        for i in range(num_layers):
            downsample = (
                PatchMerging(
                    in_channels=in_ch,
                    out_channels=2 * in_ch,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                )
                if i < num_layers - 1 else None
            )
            stage = SwinBlockSequence(
                embed_dims=in_ch,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_ch),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
            )
            self.stages.append(stage)
            if downsample:
                in_ch = downsample.out_channels

        # Output channel sizes per stage
        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]

        # Norm layer per output stage
        for i in out_indices:
            self.add_module(
                f'norm{i}',
                _build_norm_1d(norm_cfg, self.num_features[i]),
            )

        self._freeze_stages()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def out_channels(self) -> List[int]:
        return [self.num_features[i] for i in self.out_indices]

    # ── init weights ──────────────────────────────────────────────────────────

    def init_weights(self):
        if self.init_cfg is None:
            # default init
            if self.use_abs_pos_embed:
                nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            return

        # Load pretrained
        ckpt_path = self.init_cfg.get('checkpoint')
        if ckpt_path is None:
            return

        print(f"[SwinTransformer] Loading pretrained: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        state = ckpt.get('state_dict', ckpt.get('model', ckpt))

        # Strip 'backbone.' prefix if present
        state = OrderedDict(
            (k[9:] if k.startswith('backbone.') else k, v)
            for k, v in state.items()
        )
        if next(iter(state)).startswith('module.'):
            state = {k[7:]: v for k, v in state.items()}

        # Resize absolute_pos_embed if needed
        if self.use_abs_pos_embed and 'absolute_pos_embed' in state:
            ape = state['absolute_pos_embed']
            if ape.shape != self.absolute_pos_embed.shape:
                print(f"  Resizing absolute_pos_embed: {ape.shape} → {self.absolute_pos_embed.shape}")
                N, L, C = ape.shape
                H = W = int(L ** 0.5)
                H2 = W2 = int(self.absolute_pos_embed.shape[1] ** 0.5)
                ape = F.interpolate(
                    ape.reshape(N, H, W, C).permute(0, 3, 1, 2),
                    size=(H2, W2), mode='bicubic',
                ).permute(0, 2, 3, 1).reshape(N, H2 * W2, C)
                state['absolute_pos_embed'] = ape

        # Interpolate relative_position_bias_table if window size differs
        for k in [k for k in state if 'relative_position_bias_table' in k]:
            pretrained_table = state[k]
            if k in self.state_dict():
                current_table = self.state_dict()[k]
                if pretrained_table.shape != current_table.shape:
                    L1, nH = pretrained_table.shape
                    L2, _  = current_table.shape
                    S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
                    state[k] = F.interpolate(
                        pretrained_table.permute(1, 0).reshape(1, nH, S1, S1),
                        size=(S2, S2), mode='bicubic',
                    ).reshape(nH, L2).permute(1, 0).contiguous()

        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"  Missing:    {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")

    # ── freeze ────────────────────────────────────────────────────────────────

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm = getattr(self, f'norm{i - 1}')
                norm.eval()
                for p in norm.parameters():
                    p.requires_grad = False
            m = self.stages[i - 1]
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw = stage(x, hw_shape)
            if i in self.out_indices:
                norm = getattr(self, f'norm{i}')
                out  = norm(out)
                H, W = out_hw
                out  = out.view(-1, H, W, self.num_features[i])
                out  = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                outs.append(out)

        return outs


# ── Preset variants (same as mmseg) ──────────────────────────────────────────

@BACKBONES.register_module()
class SwinTransformerTiny(SwinTransformer):
    """Swin-T: embed=96, depths=(2,2,6,2), heads=(3,6,12,24)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=96, depths=(2,2,6,2), num_heads=(3,6,12,24), **kwargs)

@BACKBONES.register_module()
class SwinTransformerSmall(SwinTransformer):
    """Swin-S: embed=96, depths=(2,2,18,2), heads=(3,6,12,24)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=96, depths=(2,2,18,2), num_heads=(3,6,12,24), **kwargs)

@BACKBONES.register_module()
class SwinTransformerBase(SwinTransformer):
    """Swin-B: embed=128, depths=(2,2,18,2), heads=(4,8,16,32)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=128, depths=(2,2,18,2), num_heads=(4,8,16,32), **kwargs)

@BACKBONES.register_module()
class SwinTransformerLarge(SwinTransformer):
    """Swin-L: embed=192, depths=(2,2,18,2), heads=(6,12,24,48)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=192, depths=(2,2,18,2), num_heads=(6,12,24,48), **kwargs)