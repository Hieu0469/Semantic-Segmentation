"""
prune.py — Structured pruning EfficientViT-Seg với torch-pruning
"""

import torch
import torch.nn as nn
import torch_pruning as tp
from copy import deepcopy

from src.config import CFG


def build_ignored_layers(model: nn.Module) -> list:
    """
    Các layer không được prune:
      - output_ops (segmentation head cuối)
      - LiteMLA attention blocks
    """
    ignored = []

    # Segmentation head cuối (Conv2d → 19 classes)
    try:
        ignored.append(model.head.output_ops)
    except AttributeError:
        print("[prune] Không tìm thấy model.head.output_ops — kiểm tra lại kiến trúc")

    try:
        ignored.append(model.segmentation_head.output_ops)
    except AttributeError:
        print("[prune] Không tìm thấy model.segmentation_head.output_ops — kiểm tra lại kiến trúc")

    # LiteMLA attention blocks
    for name, module in model.named_modules():
        if "LiteMLA" in type(module).__name__:
            ignored.append(module)
            for _, sub in module.named_modules():
                if isinstance(sub, nn.Conv2d):
                    ignored.append(sub)

    return ignored


def prune_model(
    model:         nn.Module,
    pruning_ratio: float = CFG.pruning_ratio,
    round_to:      int   = 8,
    verbose:       bool  = True,
) -> nn.Module:
    """

    Parameters
    ----------
    model         : model gốc (sẽ bị deepcopy, không modify in-place)
    pruning_ratio : tỉ lệ channel bị prune (0.5 = 50%)
    round_to      : làm tròn số channel sau prune (nên dùng 8 hoặc 16)
    verbose       : in thống kê trước/sau prune

    Returns
    -------
    model đã được prune (deepcopy)
    """
    m = deepcopy(model)
    m.eval().to("cpu")

    example_inputs  = torch.randn(1, 3, CFG.train_height, CFG.train_width)
    ignored_layers  = build_ignored_layers(m)

    imp = tp.importance.GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.BasePruner(
        m,
        example_inputs,
        importance      = imp,
        pruning_ratio   = pruning_ratio,
        ignored_layers  = ignored_layers,
        iterative_steps = 1,
        isomorphic      = True,
        global_pruning  = True,
        round_to        = round_to,
    )

    if verbose:
        base_macs, base_params = tp.utils.count_ops_and_params(m, example_inputs)
        tp.utils.print_tool.before_pruning(m)

    pruner.step()

    if verbose:
        macs, params = tp.utils.count_ops_and_params(m, example_inputs)
        tp.utils.print_tool.after_pruning(m)
        print(f"MACs   : {base_macs/1e9:.2f} G  → {macs/1e9:.2f} G")
        print(f"Params : {base_params/1e6:.2f} M → {params/1e6:.2f} M")
        print(f"Giảm   : {(1 - params/base_params)*100:.1f}%")

    return m
