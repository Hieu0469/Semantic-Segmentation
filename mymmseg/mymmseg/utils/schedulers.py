"""Learning rate schedulers — build từ config dict.

Hỗ trợ:
    - PolyLR:      poly decay với warmup tùy chọn
    - CosineAnnealingLR
    - StepLR
    - MultiStepLR
    - ReduceLROnPlateau

Config style (giống mmengine):
    dict(type='PolyLR', eta_min=1e-4, power=0.9, end=40000, by_epoch=False)
"""

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


# ── PolyLR ────────────────────────────────────────────────────────────────────

class PolyLR(LambdaLR):
    """Polynomial LR decay với linear warmup tùy chọn.

    Args:
        optimizer:    Optimizer instance.
        max_steps:    Tổng số steps (hoặc epochs nếu by_epoch=True).
        eta_min:      LR tối thiểu. Default: 1e-4.
        power:        Bậc của poly. Default: 0.9.
        begin:        Step bắt đầu decay. Default: 0.
        end:          Step kết thúc (= max_steps nếu None).
        warmup_steps: Số steps warmup tuyến tính. Default: 0.
        by_epoch:     Nếu True, step tính theo epoch. Default: False.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        eta_min: float = 1e-4,
        power: float = 0.9,
        begin: int = 0,
        end: Optional[int] = None,
        warmup_steps: int = 0,
        by_epoch: bool = False,
    ):
        self.max_steps    = max_steps
        self.eta_min      = eta_min
        self.power        = power
        self.begin        = begin
        self.end          = end if end is not None else max_steps
        self.warmup_steps = warmup_steps
        self.by_epoch     = by_epoch

        # base_lrs được set bởi LambdaLR sau khi super().__init__
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        """Trả về hệ số nhân với base_lr."""
        base_lrs = self.base_lrs  # list base lr của từng param group

        # Dùng base_lr đầu tiên để tính ratio (tất cả param group scale cùng tỉ lệ)
        base_lr = base_lrs[0] if base_lrs else 1.0

        # Linear warmup
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)

        # Trước begin: giữ nguyên
        if step < self.begin:
            return 1.0

        # Sau end: clamp ở eta_min
        if step >= self.end:
            return self.eta_min / base_lr

        # Poly decay
        progress = (step - self.begin) / max(self.end - self.begin, 1)
        coeff    = (1 - progress) ** self.power
        # scale về [eta_min, base_lr]
        lr = self.eta_min + (base_lr - self.eta_min) * coeff
        return lr / base_lr


# ── Builder ───────────────────────────────────────────────────────────────────

def build_scheduler(cfg: dict, optimizer: Optimizer, max_steps: int):
    """Build scheduler từ config dict.

    Args:
        cfg:       Config dict với key 'type'.
        optimizer: Optimizer instance.
        max_steps: Tổng số steps hoặc epochs (tùy by_epoch).

    Returns:
        (scheduler, interval) — interval là 'step' hoặc 'epoch'.

    Config examples::

        # PolyLR theo step (giống mmseg)
        dict(type='PolyLR', eta_min=1e-4, power=0.9,
             begin=0, end=40000, by_epoch=False)

        # PolyLR theo epoch
        dict(type='PolyLR', eta_min=1e-6, power=0.9,
             warmup_steps=5, by_epoch=True)

        # CosineAnnealingLR
        dict(type='CosineAnnealingLR', T_max=40000, eta_min=1e-6)

        # StepLR
        dict(type='StepLR', step_size=10, gamma=0.1)

        # MultiStepLR
        dict(type='MultiStepLR', milestones=[20, 35], gamma=0.1)
    """
    import torch.optim.lr_scheduler as torch_sched

    cfg       = cfg.copy()
    sched_type = cfg.pop('type')
    by_epoch  = cfg.pop('by_epoch', True)
    interval  = 'epoch' if by_epoch else 'step'

    if sched_type == 'PolyLR':
        scheduler = PolyLR(
            optimizer,
            max_steps=max_steps,
            by_epoch=by_epoch,
            **cfg,
        )

    elif sched_type == 'CosineAnnealingLR':
        scheduler = torch_sched.CosineAnnealingLR(optimizer, **cfg)

    elif sched_type == 'StepLR':
        scheduler = torch_sched.StepLR(optimizer, **cfg)

    elif sched_type == 'MultiStepLR':
        scheduler = torch_sched.MultiStepLR(optimizer, **cfg)

    elif sched_type == 'ReduceLROnPlateau':
        scheduler = torch_sched.ReduceLROnPlateau(optimizer, **cfg)
        interval  = 'epoch'   # ReduceLROnPlateau luôn theo epoch

    elif sched_type == 'OneCycleLR':
        scheduler = torch_sched.OneCycleLR(
            optimizer, total_steps=max_steps, **cfg)
        interval = 'step'

    else:
        raise ValueError(
            f"Unknown scheduler type: '{sched_type}'. "
            f"Available: PolyLR, CosineAnnealingLR, StepLR, "
            f"MultiStepLR, ReduceLROnPlateau, OneCycleLR"
        )

    return scheduler, interval