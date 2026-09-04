"""
export_onnx.py — Export model sang ONNX và tính mIoU với ONNX Runtime

Usage:
    python export_onnx.py --model checkpoints/model.pt --output model.onnx
    python export_onnx.py --model checkpoints/model.pt --eval
"""

import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxsim import simplify

sys.path.insert(0, "efficientvit")

from src.config import CFG
from src.dataset import CityscapesDataset, ADE20KDataset, build_cityscapes_transforms, build_ade20k_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(model_path: str, output_path: str, simplify_model: bool = True):
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()

    dummy = torch.randn(1, 3, CFG.train_height, CFG.train_width)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version = 17,
        input_names   = ["input"],
        output_names  = ["output"],
        # dynamic_axes  = {
        #     "input":  {0: "batch"},
        #     "output": {0: "batch"},
        # },
        dynamo = False,
    )
    print(f"Exported → {output_path}")

    if simplify_model:
        model_onnx  = onnx.load(output_path)
        model_sim, check = simplify(model_onnx)
        assert check, "Simplify failed"
        onnx.save(model_sim, output_path)
        print(f"Simplified → {output_path}")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate ONNX mIoU
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou_onnx(
    onnx_path:    str,
    img_dir:      str,
    lbl_dir:      str,
    num_classes:  int = CFG.num_classes,
    ignore_index: int = CFG.ignore_index,
    max_samples:  int = None,
) -> dict:
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    dataset = CityscapesDataset(
        root_dir  = img_dir,
        label_dir = lbl_dir,
        transform = build_cityscapes_transforms("val"),
    )
    indices = list(range(min(max_samples, len(dataset)) if max_samples else len(dataset)))

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for idx in indices:
        image, mask = dataset[idx]
        inp    = image.unsqueeze(0).numpy().astype(np.float32)
        logits = sess.run([output_name], {input_name: inp})[0]

        # Resize logits về mask size bằng torch bicubic
        lh, lw = logits.shape[-2:]
        mh, mw = mask.shape
        if (lh, lw) != (mh, mw):
            logits = F.interpolate(
                torch.from_numpy(logits),
                size=(mh, mw), mode="bicubic", align_corners=False,
            ).numpy()

        pred  = logits[0].argmax(axis=0)
        gt    = mask.numpy()
        valid = gt != ignore_index

        combined = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
        counts   = np.bincount(combined, minlength=num_classes ** 2)
        conf_matrix += counts.reshape(num_classes, num_classes)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(indices)}] processed...")

    tp_ = np.diag(conf_matrix)
    fp_ = conf_matrix.sum(axis=0) - tp_
    fn_ = conf_matrix.sum(axis=1) - tp_

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where((tp_ + fp_ + fn_) > 0, tp_ / (tp_ + fp_ + fn_), np.nan)

    return {
        "miou":            float(np.nanmean(iou)),
        "per_class_iou":   iou,
        "per_class_names": CFG.class_names,
    }


def print_iou_report(result: dict):
    print(f"\n{'─'*45}")
    print(f"{'Class':<18} {'IoU':>8}")
    print(f"{'─'*45}")
    for name, iou in zip(result["per_class_names"], result["per_class_iou"]):
        print(f"{name:<18} {iou:.4f}" if not np.isnan(iou) else f"{name:<18}   N/A")
    print(f"{'─'*45}")
    print(f"{'mIoU':<18} {result['miou']:>8.4f}")
    print(f"{'─'*45}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path tới model .pt")
    parser.add_argument("--output", default=f"{CFG.model_name}.onnx")
    parser.add_argument("--eval",   action="store_true", help="Tính mIoU sau export")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    export_onnx(args.model, args.output)

    if args.eval:
        result = compute_iou_onnx(
            onnx_path   = args.output,
            img_dir     = f"{CFG.img_root}/val",
            lbl_dir     = f"{CFG.lbl_root}/val",
            max_samples = args.max_samples,
        )
        print_iou_report(result)
