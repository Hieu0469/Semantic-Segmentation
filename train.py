"""
train.py — Script train chính

Usage:
    python train.py                         # train từ đầu
    python train.py --resume last           # resume từ last.ckpt
    python train.py --resume checkpoints/x.ckpt
"""

import sys
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.insert(0, "efficientvit")

from src.config import CFG
from src.module import Module, CityscapesDataModule
from src.prune import prune_model


def load_base_model():
    """Load EfficientViT-L2 pretrained hoặc từ file."""
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model

    if CFG.prune_load_path is not None:
        print(f"Loading pruned model from {CFG.prune_load_path}")
        model = torch.load(CFG.prune_load_path, weights_only=False,
                           map_location="cpu")

    elif CFG.load_path is not None:
        print(f"Loading model from {CFG.load_path}")
        model = torch.load(CFG.load_path, weights_only=False, map_location="cpu")

    else:
        print("Loading pretrained EfficientViT-L2 from model zoo")
        model = create_efficientvit_seg_model(
            name        = "efficientvit-seg-l2-cityscapes",
            pretrained  = True,
            weight_url  = CFG.pretrained_url,
        )

        # Prune nếu ratio > 0
        if CFG.pruning_ratio > 0:
            print(f"Pruning model with ratio={CFG.pruning_ratio}")
            model = prune_model(model, pruning_ratio=CFG.pruning_ratio)

    return model


def main(resume: str = None):
    L.seed_everything(42, workers=True)

    # ── Model ─────────────────────────────────────────────────────────────
    CFG.model = load_base_model()

    # ── Logger ────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir = CFG.log_dir,
        name     = CFG.model_name,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath    = CFG.ckpt_dir,
        filename   = CFG.model_name + "-ep{epoch:03d}-miou{val/mIoU:.4f}",
        monitor    = "val/mIoU",
        mode       = "max",
        save_top_k = 3,
        save_last  = True,
        auto_insert_metric_name = False,
    )
    early_stop_cb = EarlyStopping(
        monitor   = "val/mIoU",
        patience  = 15,
        mode      = "max",
        min_delta = 0.001,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs              = CFG.max_epochs,
        accelerator             = "auto",
        devices                 = "auto",
        precision               = "16-mixed",
        logger                  = logger,
        callbacks               = [checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps       = 10,
        gradient_clip_val       = 1.0,
        check_val_every_n_epoch = 1,
        num_sanity_val_steps    = 2,
    )

    # ── Fit ───────────────────────────────────────────────────────────────
    ckpt_path = None
    if resume == "last":
        ckpt_path = f"{CFG.ckpt_dir}/last.ckpt"
    elif resume is not None:
        ckpt_path = resume

    lightning_model = Module(CFG)
    dm = CityscapesDataModule(CFG)

    trainer.fit(lightning_model, datamodule=dm, ckpt_path=ckpt_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nBest checkpoint : {checkpoint_cb.best_model_path}")
    score = checkpoint_cb.best_model_score
    print(f"Best val mIoU   : {score:.4f}" if score is not None
          else "Best val mIoU   : N/A")

    # ── Validate best ─────────────────────────────────────────────────────
    results = trainer.validate(
        lightning_model,
        datamodule = dm,
        ckpt_path  = checkpoint_cb.best_model_path,
    )
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="'last' hoặc path tới .ckpt để resume")
    args = parser.parse_args()
    main(resume=args.resume)
