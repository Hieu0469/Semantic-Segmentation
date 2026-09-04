"""
visualize.py — Visualize kết quả segmentation trên test set

Usage:
    python visualize.py --model checkpoints/model.pt --n 20
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "efficientvit")

from src.config import CFG
from src.dataset import CityscapesDataset, build_transforms, UnNormalize, CITYSCAPES_PALETTE


def visualize(model_path: str, n_samples: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    test_dataset = CityscapesDataset(
        root_dir  = f"{CFG.img_root}/test",
        transform = build_transforms("val"),
    )

    unorm = UnNormalize()

    for i in range(min(n_samples, len(test_dataset))):
        x = test_dataset[i]
        if isinstance(x, (list, tuple)):
            x = x[0]

        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device))
            if logits.shape[-2:] != x.shape[-2:]:
                import torch.nn.functional as F
                logits = F.interpolate(logits, size=x.shape[-2:],
                                       mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).squeeze().cpu().numpy()

        color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for cls_id, color in enumerate(CITYSCAPES_PALETTE):
            color_mask[pred == cls_id] = color

        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(unorm(x).permute(1, 2, 0).clamp(0, 1))
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(color_mask)
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"viz_{i:03d}.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved viz_{i:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path tới model .pt")
    parser.add_argument("--n",     type=int, default=20, help="Số ảnh visualize")
    args = parser.parse_args()
    visualize(args.model, args.n)
