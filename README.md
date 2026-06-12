
# EfficientViT & SMP Semantic Segmentation (Jetson AGX Xavier Deploy Ready)

This repository provides a flexible and modular pipeline to train, structurally prune, and deploy semantic segmentation models (including **EfficientViT** and **Segmentation Models PyTorch - SMP**) on the **NVIDIA Jetson AGX Xavier** platform using **PyTorch Lightning**.

---

## 📂 Repository Structure
```text
Semantic-Segmentation/
├── configs/
│   └── config.py          # Centralized configuration (Hyperparameters, Model Flags, Paths)
├── data/
│   └── dataset.py         # Data processing, Albumentations, and DataModule pipelines
├── models/
│   ├── model.py           # Core model initialization & torch_pruning execution logic
│   └── lit_module.py      # PyTorch Lightning training/validation steps & loss modules
├── train.py               # Main pipeline orchestrator (Init -> Prune -> Train)
├── inference.py           # Model evaluation and visual result plotting script
└── requirements.txt       # Dependencies list

```

---

## 🛠️ Quick Start

### 1. Environment Setup

```bash
git clone [https://github.com/Hieu0469/Semantic-Segmentation.git](https://github.com/Hieu0469/Semantic-Segmentation.git)
cd Semantic-Segmentation
pip install -r requirements.txt

# (Optional) If using EfficientViT
git clone [https://github.com/mit-han-lab/efficientvit.git](https://github.com/mit-han-lab/efficientvit.git)
pip install -e efficientvit/

```

### 2. Pre-trained Models Download

You can download the pre-trained backbone weights and fine-tuned checkpoints from Google Drive storage:

👉[DOWNLOAD PRE-TRAINED MODELS HERE](https://drive.google.com/drive/folders/1OW-J-ApF7Z93v9zSZ0Ftp4F2Fg9PKJIt?usp=drive_link )

(Place the downloaded `.pth` or `.pt` files into the `checkpoints/` directory before training or testing).

👉[DOWNLOAD ONNX MODELS HERE](https://drive.google.com/drive/u/0/folders/1b9rk17gkG6b_a8rtO_dX2yixcUKfhQDs)


## ⚙️ Configuration Setup (CFG)

All settings are configured inside `configs/config.py`. Below is the blueprint setup. Make sure to define your target paths and model selection flags:

```python
import os
from src.dataset import CITYSCAPES_CLASS_NAMES, CITYSCAPES_CLASS_WEIGHTS

class CFG:
    # ── Model Selection ───────────────────────────────────────────────────
    model_name       = "Resnet18_DeeplabV3Plus"
    model_type       = 'smp'               # Choose: 'efficientvit' or 'smp'
    smp_architecture = 'DeepLabV3Plus'     # Options: Unet, UnetPlusPlus, PSPNet, DeepLabV3Plus, etc.
    smp_encoder      = 'resnet18'          # Choose any backbone supported by SMP
    pretrained       = True
    num_classes      = 19
    ignore_index     = 255
    model            = None                # Accepts a direct PyTorch Model Object injection

    # ── Pruning Configuration  ────────────────────────────────────────────
    use_pruning      = False               # Toggle structural channel pruning
    pruning_ratio    = 0.5                 # 50% target structural reduction

    # ── Data Paths ────────────────────────────────────────────────────────
    data_root  = "./dataset/Cityscape Dataset"
    label_root = "./dataset/Fine Annotations"
    img_root   = os.path.join(data_root,  "leftImg8bit")
    lbl_root   = os.path.join(label_root, "gtFine")
    log_dir    = "tb_logs"
    ckpt_dir   = "checkpoints"

    # ── Hyperparameters ───────────────────────────────────────────────────
    train_height, train_width = 512, 1024
    batch_size    = 4
    num_workers   = 4
    max_epochs    = 100
    lr            = 1e-4
    weight_decay  = 1e-4
    warmup_epochs = 5

    class_names   = CITYSCAPES_CLASS_NAMES
    class_weights = CITYSCAPES_CLASS_WEIGHTS

```

---

## 🏋️ Training Execution

The pipeline automatically handles initialization, pruning logic (if activated), and PyTorch Lightning routing based on your `CFG` choices.

To kick off the training process, run:

```bash
python train.py

```

* **EfficientViT Path:** If `use_pruning = True`, `torch_pruning` will dynamically shrink channels by 50% using $L_2$-norm group evaluation while safeguarding `LiteMLA` blocks before training.
* **SMP Path:** Automatically downloads requested ImageNet weights and initializes the segmentation architecture seamlessly.


Dưới đây là toàn bộ nội dung file `README.md` được viết dưới dạng văn bản thô (raw text) bên trong khối mã nguồn. Tôi đã cập nhật thêm phần link tham khảo trực tiếp (Hyperlinks) đến hai kho lưu trữ mã nguồn chính thức của **EfficientViT** và **Segmentation Models PyTorch (SMP)** ở mục tài liệu tham khảo phía cuối file để tăng độ uy tín cho repository của bạn.

Bạn chỉ cần nhấn nút sao chép toàn bộ khối mã này và dán đè vào file `README.md` trên GitHub:

```text
# EfficientViT & SMP Semantic Segmentation (Jetson AGX Xavier Deploy Ready)

This repository provides a flexible and modular pipeline to train, structurally prune, and deploy semantic segmentation models (including **EfficientViT** and **Segmentation Models PyTorch - SMP**) on the **NVIDIA Jetson AGX Xavier** platform using **PyTorch Lightning**.

---

## 📂 Repository Structure
```text
Semantic-Segmentation/
├── configs/
│   └── config.py          # Centralized configuration (Hyperparameters, Model Flags, Paths)
├── data/
│   └── dataset.py         # Data processing, Albumentations, and DataModule pipelines
├── models/
│   ├── model.py           # Core model initialization & torch_pruning execution logic
│   └── lit_module.py      # PyTorch Lightning training/validation steps & loss modules
├── train.py               # Main pipeline orchestrator (Init -> Prune -> Train)
├── inference.py           # Model evaluation and visual result plotting script
└── requirements.txt       # Dependencies list

```

---

## 🛠️ Quick Start

### 1. Environment Setup

```bash
git clone [https://github.com/Hieu0469/Semantic-Segmentation.git](https://github.com/Hieu0469/Semantic-Segmentation.git)
cd Semantic-Segmentation
pip install -r requirements.txt

# (Optional) If using EfficientViT
git clone [https://github.com/mit-han-lab/efficientvit.git](https://github.com/mit-han-lab/efficientvit.git)
pip install -e efficientvit/

```

### 2. Pre-trained Models Download

You can download the pre-trained backbone weights and fine-tuned checkpoints from our Google Drive storage:
👉 **[DOWNLOAD PRE-TRAINED MODELS & CHECKPOINTS HERE](👉 CHÈN_LINK_GOOGLE_DRIVE_CỦA_BẠN_VÀO_ĐÂY 👈)**
*(Place the downloaded `.pth` or `.pt` files into the `checkpoints/` directory before training or testing).*

---

## ⚙️ Configuration Setup (CFG)

All settings are configured inside `configs/config.py`. Below is the blueprint setup. Make sure to define your target paths and model selection flags:

```python
import os
from src.dataset import CITYSCAPES_CLASS_NAMES, CITYSCAPES_CLASS_WEIGHTS

class CFG:
    # ── Model Selection ───────────────────────────────────────────────────
    model_name       = "Resnet18_DeeplabV3Plus"
    model_type       = 'smp'               # Choose: 'efficientvit' or 'smp'
    smp_architecture = 'DeepLabV3Plus'     # Options: Unet, UnetPlusPlus, PSPNet, DeepLabV3Plus, etc.
    smp_encoder      = 'resnet18'          # Choose any backbone supported by SMP
    pretrained       = True
    num_classes      = 19
    ignore_index     = 255
    model            = None                # Accepts a direct PyTorch Model Object injection

    # ── Pruning Configuration (EfficientViT Only) ─────────────────────────
    use_pruning      = False               # Toggle structural channel pruning
    pruning_ratio    = 0.5                 # 50% target structural reduction

    # ── Data Paths ────────────────────────────────────────────────────────
    data_root  = "./dataset/Cityscape Dataset"
    label_root = "./dataset/Fine Annotations"
    img_root   = os.path.join(data_root,  "leftImg8bit")
    lbl_root   = os.path.join(label_root, "gtFine")
    log_dir    = "tb_logs"
    ckpt_dir   = "checkpoints"

    # ── Hyperparameters ───────────────────────────────────────────────────
    train_height, train_width = 512, 1024
    batch_size    = 4
    num_workers   = 4
    max_epochs    = 100
    lr            = 1e-4
    weight_decay  = 1e-4
    warmup_epochs = 5

    class_names   = CITYSCAPES_CLASS_NAMES
    class_weights = CITYSCAPES_CLASS_WEIGHTS

```

---

## 🏋️ Training Execution

The pipeline automatically handles initialization, pruning logic (if activated), and PyTorch Lightning routing based on your `CFG` choices.

To kick off the training process, run:

```bash
python train.py

```

* **EfficientViT Path:** If `use_pruning = True`, `torch_pruning` will dynamically shrink channels by 50% using $L_2$-norm group evaluation while safeguarding `LiteMLA` blocks before training.
* **SMP Path:** Automatically downloads requested ImageNet weights and initializes the segmentation architecture seamlessly.

---



## 📜 References & Acknowledgments

* **EfficientViT Core Repository:** [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) — Official implementation of high-resolution hardware-efficient vision models.
* **Segmentation Models PyTorch:** [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) — Python library with Neural Networks for Image Segmentation based on PyTorch.
* **Torch-Pruning Toolkit:** [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning) — Structural pruning library for deep networks.
* **PyTorch Lightning Framework:** [Lightning-AI/lightning](https://github.com/Lightning-AI/lightning) — Lightweight PyTorch wrapper for high-performance AI research.
