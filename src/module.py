"""
module.py — LightningModule và DataModule cho EfficientViT-Seg
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from torchmetrics import JaccardIndex
from segmentation_models_pytorch.losses import DiceLoss

from src.config import CFG
from src.dataset import CityscapesDataset, build_transforms

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class CityscapesDataModule(L.LightningDataModule):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        for split, attr in [("train", "train_ds"), ("val", "val_ds")]:
            setattr(self, attr, CityscapesDataset(
                root_dir     = os.path.join(self.cfg.img_root, split),
                label_dir    = os.path.join(self.cfg.lbl_root, split),
                transform    = build_transforms(split),
                train_height = self.cfg.train_height,
                train_width  = self.cfg.train_width,
            ))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = True,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class Module(L.LightningModule):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.model = cfg.model

        # ── Loss ──────────────────────────────────────────────────────────
        weights = torch.tensor(cfg.class_weights, dtype=torch.float32)
        self.ce_loss   = nn.CrossEntropyLoss(
            weight       = weights,
            ignore_index = cfg.ignore_index,
        )
        self.dice_loss = DiceLoss(
            mode         = "multiclass",
            ignore_index = cfg.ignore_index,
            from_logits  = True,
        )
        self.dice_weight = 0.5

        # ── Metrics ───────────────────────────────────────────────────────
        metric_kwargs = dict(
            task         = "multiclass",
            num_classes  = cfg.num_classes,
            ignore_index = cfg.ignore_index,
            average      = "macro",
        )
        self.train_miou = JaccardIndex(**metric_kwargs)
        self.val_miou   = JaccardIndex(**metric_kwargs)
        self.best_miou  = 0.0

    # ── helpers ──────────────────────────────────────────────────────────
    def _resize_to_mask(self, logits, masks):
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return logits

    def _compute_loss(self, logits, masks):
        ce   = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        return ce + self.dice_weight * dice, ce, dice

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        return self.model(x)

    # ── training ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self._resize_to_mask(self(images), masks)
        loss, ce, dice = self._compute_loss(logits, masks)

        preds = logits.argmax(dim=1)
        self.train_miou(preds, masks)

        self.log_dict({
            "train/loss":      loss,
            "train/ce_loss":   ce,
            "train/dice_loss": dice,
            "train/mIoU":      self.train_miou,
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ── validation ───────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self._resize_to_mask(self(images), masks)
        loss, ce, dice = self._compute_loss(logits, masks)

        preds = logits.argmax(dim=1)
        self.val_miou(preds, masks)

        self.log_dict({
            "val/loss":      loss,
            "val/ce_loss":   ce,
            "val/dice_loss": dice,
            "val/mIoU":      self.val_miou,
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self._log_seg_image(images[0], masks[0], preds[0])

    def _log_seg_image(self, image, gt_mask, pred_mask):
        import numpy as np
        gt_rgb   = CityscapesDataset.decode_segmap(gt_mask)
        pred_rgb = CityscapesDataset.decode_segmap(pred_mask)
        vis = np.concatenate([gt_rgb, pred_rgb], axis=1)
        vis = torch.from_numpy(vis).permute(2, 0, 1).float() / 255.0
        self.logger.experiment.add_image(
            "val/gt_vs_pred", vis, global_step=self.current_epoch
        )

    # ── predict ──────────────────────────────────────────────────────────
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, masks = batch
        else:
            images, masks = batch, None

        logits = self(images)
        if masks is not None and logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)

        results = []
        for i in range(len(images)):
            item = {"pred_rgb": CityscapesDataset.decode_segmap(preds[i])}
            if masks is not None:
                item["gt_rgb"] = CityscapesDataset.decode_segmap(masks[i])
            results.append(item)
        return results

    # ── epoch end ────────────────────────────────────────────────────────
    def on_validation_epoch_end(self):
        confmat = self.val_miou.confmat.cpu().numpy()
        tp_  = np.diag(confmat)
        fp_  = confmat.sum(axis=0) - tp_
        fn_  = confmat.sum(axis=1) - tp_

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where((tp_ + fp_ + fn_) > 0, tp_ / (tp_ + fp_ + fn_), np.nan)

        miou = float(np.nanmean(iou))

        # ── Log per-class IoU lên TensorBoard ────────────────────────────
        for name, v in zip(self.cfg.class_names, iou):
            if not np.isnan(v):
                self.log(f"val/iou_{name}", float(v), sync_dist=True)

        # ── Build log string ──────────────────────────────────────────────
        lines = []
        lines.append(f"\nEpoch {self.current_epoch:03d}  |  mIoU: {miou:.4f}")
        lines.append("─" * 40)
        for name, v in zip(self.cfg.class_names, iou):
            lines.append(f"  {name:<18} {v:.4f}" if not np.isnan(v) else f"  {name:<18}   N/A")
        lines.append("─" * 40)
        log_str = "\n".join(lines)

        # ── Ghi val_results.txt (append) ─────────────────────────────────
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        with open(os.path.join(self.cfg.log_dir, "val_results.txt"), "a") as f:
            f.write(log_str + "\n")

        # ── Lưu best model (.pt) ─────────────────────────────────────────
        if miou > self.best_miou:
            self.best_miou = miou
            os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

            # Xoá file best cũ
            for fname in os.listdir(self.cfg.ckpt_dir):
                if fname == f"{self.cfg.model_name}.pt":
                    os.remove(os.path.join(self.cfg.ckpt_dir, fname))

            save_path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.model_name}.pt")
            torch.save(self.model, save_path)
            tqdm.write(f"\n✓ Saved best model → {save_path}  (mIoU: {miou:.4f})\n")

            # Ghi best_val.txt (ghi đè)
            with open(os.path.join(self.cfg.log_dir, "best_val.txt"), "w") as f:
                f.write(f"Epoch     : {self.current_epoch:03d}\n")
                f.write(f"mIoU      : {miou:.6f}\n")
                f.write(log_str + "\n")

    # ── optimizer + scheduler ────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr           = self.cfg.lr,
            weight_decay = self.cfg.weight_decay,
        )

        total  = self.cfg.max_epochs
        warmup = self.cfg.warmup_epochs

        def poly_with_warmup(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total - warmup, 1)
            return (1 - progress) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_with_warmup)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
