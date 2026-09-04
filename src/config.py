"""
config.py — Tất cả hyperparameter và đường dẫn tập trung ở đây.
Chỉnh sửa file này trước khi train.
"""

import os
from src.dataset import CITYSCAPES_CLASS_NAMES, CITYSCAPES_CLASS_WEIGHTS

class CFG:
    # ── Model ─────────────────────────────────────────────────────────────
    model_name   = "EfficientVitL2_city"
    model_type = 'efficientvit'               # Chọn: 'efficientvit' hoặc 'smp'
    num_classes  = 19
    ignore_index = 255
    model = None  # Nếu muốn truyền trực tiếp một PyTorch Model Object, gán ở đây (ví dụ: model = MyCustomModel())
    
    smp_architecture = None       # Chọn kiến trúc: Unet, UnetPlusPlus, PSPNet, DeepLabV3Plus, v.v.
    smp_encoder = None     # Chọn backbone encoder tùy ý của thư viện SMP
    dataset_name   = "cityscapes"
    efficientvit_variant = "l2"  # Chọn variant của EfficientViT: b0, b1, b2, b3, l1, l2
    pretrained = False
    # ── Pretrained / checkpoint ───────────────────────────────────────────
    pretrained_url   = None   # URL hoặc path tới weights gốc EfficientViT-L2
    load_path        = None   # Path tới model đã train (.pt) để tiếp tục train
    checkpoint_path  = None   # Lightning .ckpt để resume training

    # ── Pruning ───────────────────────────────────────────────────────────
    use_pruning   = False  # Có sử dụng pruning không
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
    lr            = 1e-4
    weight_decay  = 1e-4
    warmup_epochs = 5

    # ── Class names (Cityscapes 19 classes) ───────────────────────────────
    class_names = CITYSCAPES_CLASS_NAMES

    # ── Class weights cho weighted CrossEntropy ───────────────────────────
    # class_weights = CITYSCAPES_CLASS_WEIGHTS