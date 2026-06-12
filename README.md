

# EfficientViT-L2 Semantic Segmentation with Structural Pruning for Jetson AGX Xavier

## 📝 Introduction
This project focuses on deploying a high-performance, real-time semantic segmentation solution on the **NVIDIA Jetson AGX Xavier** edge platform. Edge devices typically face strict computational and memory constraints, making the deployment of heavy deep learning models challenging. To bridge this gap, this repository provides an end-to-end pipeline that leverages **EfficientViT-L2** trained on the **Cityscapes dataset**, optimized via **PyTorch Lightning**, and compressed using **torch_pruning**. 

By applying structural channel pruning, we achieve a **50% reduction** in model parameters and significantly lower computational complexity. This structural optimization ensures a lightweight footprint and faster inference speeds while preserving high segmentation accuracy (mIoU), making the model ideal for autonomous driving research and real-time environment perception on embedded systems.


## 🛠️ Environment Setup

### Prerequisites
- **Host / Target OS:** Ubuntu 20.04 LTS or 22.04 LTS
- **NVIDIA JetPack:** v5.x or v6.x (which includes pre-configured CUDA, cuDNN, and TensorRT environments)
- **Python:** >= 3.8

### Installation

1. **Clone the repository:**
 ```bash
 git clone [https://github.com/Hieu0469/Semantic-Segmentation.git](https://github.com/Hieu0469/Semantic-Segmentation.git)
 cd Semantic-Segmentation

  ```

2. **Install core packages:**
```bash
pip install -r requirements.txt

```


3. **Install the EfficientViT core library:**
```bash
git clone [https://github.com/mit-han-lab/efficientvit.git](https://github.com/mit-han-lab/efficientvit.git)
pip install -e efficientvit/

```



---

## 📊 Dataset Preparation

The pipeline is structured to train on the official **Cityscapes Dataset**. You need to download both the **LeftImg8bit** (images) and **Fine Annotations** (labels) packages. Extract and organize your local dataset tree exactly as shown below:

```text
dataset/
├── Cityscape Dataset/
│   └── leftImg8bit/
│       ├── train/
│       └── val/
└── Fine Annotations/
    └── gtFine/
        ├── train/
        └── val/

```

---

## ⚙️ Configuration Setup (CFG)

All structural properties, model hyperparameters, and dataset paths are managed centrally inside `configs/config.py`. Before initiating a training session, modify the directory strings and adjust target parameters inside this configuration block.

Here is the production blueprint for your `configs/config.py`:

```python
import os

class CFG:
    # Model Specs
    model_name = '0.5pruned_EfficientVitL2_city'
    num_classes = 19
    train_width = 1024
    train_height = 512
    
    # Dataset Paths
    data_root = "./dataset/Cityscape Dataset"
    label_root = "./dataset/Fine Annotations"
    img_root  = os.path.join(data_root, "leftImg8bit")
    lbl_root  = os.path.join(label_root, "gtFine")
    
    # Outputs & Profiling
    log_dir   = "tb_logs"
    ckpt_dir  = "checkpoints"
 
    # Hyperparameters
    ignore_index = 255
    batch_size   = 4
    num_workers  = 4
    n_eps = 100
    max_epochs   = n_eps
    learning_rate = 1e-5
    lr           = learning_rate
    weight_decay = 1e-4
    warmup_epochs = 5
    
    # Compression Setup
    pruning_ratio = 0.5  # 50% target structural channel reduction
    
    class_names = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle",
    ]

```

---

## 🏋️ Pruning & Training Workflow

The training logic integrates model compression directly before fine-tuning. When executing the pipeline, the sequence progresses through two fundamental phases:

### 1. Model Initialization & Structural Pruning

Before updating any weights, `models/model.py` intercepts the pre-trained base model. By passing an initial dummy tensor through the network, `torch_pruning` dynamically maps out dependency groups across individual layers:

* **Importance Evaluation:** A group-level $L_2$-norm metric (`GroupMagnitudeImportance`) analyzes channel weight magnitudes to determine less critical groups.
* **Layer Safeguarding:** The pruner explicitly ignores the final classification layers (`head.output_ops`) and internal multi-scale lightweight attention layers (`LiteMLA`). This preserves the integrity of essential spatial relation operations, preventing critical hardware compatibility breaks or accuracy drops.

### 2. Execution Call

To initiate the structural channel reduction and immediately launch the fine-tuning schedule on the compressed network, run:

```bash
python train.py

```

During execution, **PyTorch Lightning** orchestrates the workflow:

* Activates `16-mixed` precision to optimize memory footprints on Tensor cores.
* Computes cross-entropy loss while ignoring unlabeled boundary indices (`255`).
* Evaluates Mean Intersection over Union (`val/mIoU`) after every epoch, automatically saving the best model states in the `checkpoints/` folder.

---

## 🖥️ Evaluation & Analytics

To validate the model's performance, compute class-wise Intersection over Union metrics, and generate qualitative segmentation visualizations, run the inference script:

```bash
python inference.py

```

### Empirical Evaluation Baseline

Upon completing the fine-tuning phase on the 50% pruned EfficientViT-L2 network, the model yields the following IoU distribution across the validation split:

| Target Class | Measured IoU | Target Class | Measured IoU |
| --- | --- | --- | --- |
| **Road** | ~0.9774 | **Sidewalk** | ~0.8205 |
| **Building** | ~0.9062 | **Wall** | ~0.4500 |
| **Fence** | ~0.5200 | **Pole** | ~0.6100 |
| **Traffic Light** | ~0.7100 | **Traffic Sign** | ~0.7800 |
| **Vegetation** | ~0.9150 | **Sky** | ~0.9300 |

---

## 🔮 NVIDIA Jetson AGX Xavier Deployment

To fully utilize the specialized hardware accelerators on the Jetson board, follow this serialization sequence to compile your model:

### Step 1: Export to Static ONNX

Serialize your refined `.pth` checkpoint into a static graph format:

```bash
python export_onnx.py --checkpoint checkpoints/0.5pruned_EfficientVitL2_city.pth --output efficientvit_pruned.onnx

```

### Step 2: Optimize Graph Complexity

Run `onnx-simplifier` to fuse redundant nodes, eliminate dead branches, and execute constant folding:

```bash
onnxsim efficientvit_pruned.onnx efficientvit_pruned_sim.onnx

```

### Step 3: Compile TensorRT Runtime Engine on Jetson

Transfer your `efficientvit_pruned_sim.onnx` asset directly to the filesystem of your **NVIDIA Jetson AGX Xavier**. Run the native `trtexec` utility to build a highly optimized runtime engine using half-precision floating-point optimizations (`FP16`):

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=efficientvit_pruned_sim.onnx \
  --saveEngine=efficientvit_pruned_fp16.engine \
  --fp16 \
  --workspace=4096 \
  --verbose

```

---

## 📜 References & Acknowledgments

* **EfficientViT:** [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
* **Torch-Pruning:** [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
* **PyTorch Lightning:** [Lightning-AI/lightning](https://github.com/Lightning-AI/lightning)

