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
from src.dataset import ADE20KDataset, CityscapesDataset, build_ade20k_transforms, build_cityscapes_transforms
from export_onnx import export_onnx
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DataModule - CityscapesDataset
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
                transform    = build_cityscapes_transforms(split, self.cfg.train_height, self.cfg.train_width),
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
# DataModule — ADE20K
# ─────────────────────────────────────────────────────────────────────────────
class ADE20KDataModule(L.LightningDataModule):
    """
    Yêu cầu trong cfg:
        cfg.ade20k_root  → path tới ADEChallengeData2016
                            (chứa images/ và annotations/)
    Cấu trúc:
        {ade20k_root}/images/training,      images/validation
        {ade20k_root}/annotations/training, annotations/validation
    """

    SPLIT_FOLDER = {"train": "training", "val": "validation"}

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        for split, attr in [("train", "train_ds"), ("val", "val_ds")]:
            folder = self.SPLIT_FOLDER[split]
            setattr(self, attr, ADE20KDataset(
                img_dir      = os.path.join(self.cfg.ade20k_root, "images", folder),
                label_dir    = os.path.join(self.cfg.ade20k_root, "annotations", folder),
                transform    = build_ade20k_transforms(split, self.cfg.train_height, self.cfg.train_width),
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
class SegmentationModule(L.LightningModule):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.model = cfg.model
        cfg.model.train()

        dataset_name = cfg.dataset_name.lower()
        if dataset_name == "cityscapes":
            self._decode_segmap = CityscapesDataset.decode_segmap
        elif dataset_name == "ade20k":
            self._decode_segmap = ADE20KDataset.decode_segmap
        else:
            raise ValueError(f"dataset_name '{cfg.dataset_name}' không hợp lệ. Hãy chọn 'cityscapes' hoặc 'ade20k'.")
        # ── Loss ──────────────────────────────────────────────────────────
        weights = getattr(cfg, "class_weights", None)
        weights = torch.tensor(weights, dtype=torch.float) if weights is not None else None
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
        return ce + self.dice_weight * dice

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        return self.model(x)

    # ── training ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)                       # (B, C, H, W)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],   # (H, W)
                mode="bilinear",
                align_corners=False,
            )
        loss = self._compute_loss(logits, masks)
 
        preds = logits.argmax(dim=1)
        self.train_miou(preds, masks)
 
        self.log("train/loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mIoU", self.train_miou,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    # ── validation ───────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],   # (H, W)
                mode="bilinear",
                align_corners=False,
            )
        loss = self._compute_loss(logits, masks)
        preds  = logits.argmax(dim=1)
        self.val_miou(preds, masks)
    
        self.log("val/loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mIoU", self.val_miou,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log ảnh segmentation đầu tiên của mỗi epoch
        if batch_idx == 0:
            self._log_seg_image(images[0], masks[0], preds[0])

    def _log_seg_image(self, image, gt_mask, pred_mask):
        import numpy as np
        gt_rgb   = self._decode_segmap(gt_mask)
        pred_rgb = self._decode_segmap(pred_mask)
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
            item = {"pred_rgb": self._decode_segmap(preds[i])}
            if masks is not None:
                item["gt_rgb"] = self._decode_segmap(masks[i])
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
        
            save_path = os.path.join(
                self.cfg.ckpt_dir,
                f"{self.cfg.model_name}.pt"
            )
            torch.save(self.model, save_path)
            export_onnx(save_path, os.path.join(self.cfg.ckpt_dir, f"{self.cfg.model_name}.onnx"))
            tqdm.write(f"\n✓ Saved best model → {save_path}  (mIoU: {miou:.4f})\n")
            # Log best val
    
            # Ghi đè file — luôn chỉ giữ kết quả tốt nhất
            log_path = os.path.join(self.cfg.log_dir, "best_val.txt")
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_str + "\n")
    
            tqdm.write(f"\n✓ New best mIoU: {miou:.4f} → saved to {log_path}\n")
            

    # ── optimizer + scheduler ────────────────────────────────────────────
    def configure_optimizers(self):
        if self.cfg.model_type == "efficientvit":
            encoder_params = list(self.model.backbone.parameters())
            decoder_params = list(self.model.head.parameters())
        elif self.cfg.model_type == "smp":
            encoder_params = list(self.model.encoder.parameters())
            decoder_params = list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())
        else:
            raise ValueError(f"model_type '{self.cfg.model_type}' không hợp lệ. Hãy chọn 'efficientvit' hoặc 'smp'.")
        
        
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.cfg.lr * 0.1},
            {"params": decoder_params, "lr": self.cfg.lr},
        ], weight_decay=self.cfg.weight_decay)
    
        # Poly LR với linear warmup
        total   = self.cfg.max_epochs
        warmup  = self.cfg.warmup_epochs
    
        def poly_with_warmup(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total - warmup, 1)
            return (1 - progress) ** 0.9
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_with_warmup)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
