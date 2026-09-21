"""Optimizer builder — build từ config dict.

Config style:
    dict(type='AdamW', lr=6e-5, weight_decay=0.01)
    dict(type='SGD',   lr=0.01, momentum=0.9, weight_decay=5e-4)
    dict(type='Adam',  lr=1e-4)
"""

from typing import List, Optional
import torch.optim as optim


def build_optimizer(cfg: dict, params) -> optim.Optimizer:
    """Build optimizer từ config dict.

    Args:
        cfg:    Config dict với key 'type' và các hyperparams.
        params: Model parameters hoặc list param groups.

    Returns:
        Optimizer instance.

    Config examples::

        dict(type='AdamW', lr=6e-5, weight_decay=0.01)
        dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4,
             nesterov=True)
        dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
        dict(type='RMSprop', lr=1e-3, momentum=0.9)
    """
    cfg        = cfg.copy()
    optim_type = cfg.pop('type')

    if optim_type == 'AdamW':
        return optim.AdamW(params, **cfg)
    elif optim_type == 'Adam':
        return optim.Adam(params, **cfg)
    elif optim_type == 'SGD':
        return optim.SGD(params, **cfg)
    elif optim_type == 'RMSprop':
        return optim.RMSprop(params, **cfg)
    elif optim_type == 'Adamax':
        return optim.Adamax(params, **cfg)
    else:
        raise ValueError(
            f"Unknown optimizer type: '{optim_type}'. "
            f"Available: AdamW, Adam, SGD, RMSprop, Adamax"
        )