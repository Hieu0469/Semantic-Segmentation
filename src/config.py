"""
config.py — Tất cả hyperparameter và đường dẫn tập trung ở đây.
Chỉnh sửa file này trước khi train.
"""

import os


class CFG:
    # ── Model ─────────────────────────────────────────────────────────────
    model_name   = "0.5pruned_EfficientVitL2_city"
    num_classes  = 19
    ignore_index = 255

    # ── Pretrained / checkpoint ───────────────────────────────────────────
    pretrained_url   = None   # URL hoặc path tới weights gốc EfficientViT-L2
    load_path        = None   # Path tới model đã train (.pt) để tiếp tục train
    prune_load_path  = None   # Path tới model đã prune để load và train
    checkpoint_path  = None   # Lightning .ckpt để resume training

    # ── Pruning ───────────────────────────────────────────────────────────
    pruning_ratio = 0.5       # Tỉ lệ prune channel (0.0 = không prune)

    # ── Paths ─────────────────────────────────────────────────────────────
    data_root  = "/kaggle/input/datasets/electraawais/cityscape-dataset/Cityscape Dataset"
    label_root = "/kaggle/input/datasets/electraawais/cityscape-dataset/Fine Annotations"
    img_root   = os.path.join(data_root,  "leftImg8bit")
    lbl_root   = os.path.join(label_root, "gtFine")
    log_dir    = "tb_logs"
    ckpt_dir   = "checkpoints"

    # ── Input ─────────────────────────────────────────────────────────────
    train_height = 512
    train_width  = 1024

    # ── Training ──────────────────────────────────────────────────────────
    batch_size    = 4
    num_workers   = 4
    max_epochs    = 100
    lr            = 1e-5
    weight_decay  = 1e-4
    warmup_epochs = 5

    # ── Class names (Cityscapes 19 classes) ───────────────────────────────
    class_names = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle",
    ]

    # ── Class weights cho weighted CrossEntropy ───────────────────────────
    class_weights = [
        0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
        1.0865, 1.0955, 1.0865, 1.1529, 1.0507,
    ]
