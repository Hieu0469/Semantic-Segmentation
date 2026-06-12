# EfficientViT-L2 Cityscapes Segmentation

Structured pruning + fine-tuning của **EfficientViT-Seg-L2** trên **Cityscapes** với PyTorch Lightning.

## Cấu trúc project

```
├── src/
│   ├── config.py       # Tất cả hyperparameter và đường dẫn
│   ├── dataset.py      # CityscapesDataset + augmentations
│   ├── prune.py        # Structured pruning với torch-pruning
│   └── module.py       # LightningModule + DataModule
├── train.py            # Script train chính
├── export_onnx.py      # Export ONNX + tính mIoU
├── visualize.py        # Visualize kết quả trên test set
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
# Clone repo và các dependency
git clone https://github.com/your-username/efficientvit-cityscapes
cd efficientvit-cityscapes

git clone https://github.com/mit-han-lab/efficientvit.git
git clone https://github.com/Coder0469/MyTorchPruning.git

pip install -r requirements.txt
```

## Dataset

Cấu trúc Cityscapes:
```
data/
  leftImg8bit/
    train/  val/  test/
  gtFine/
    train/  val/  test/
```

Cập nhật đường dẫn trong `src/config.py`:
```python
class CFG:
    data_root  = "/path/to/leftImg8bit"
    label_root = "/path/to/gtFine"
```

## Train

```bash
# Train từ đầu (tự prune nếu pruning_ratio > 0)
python train.py

# Resume từ last checkpoint
python train.py --resume last

# Resume từ checkpoint cụ thể
python train.py --resume checkpoints/model-ep050.ckpt
```

## Export ONNX

```bash
# Export
python export_onnx.py --model checkpoints/model.pt --output model.onnx

# Export + tính mIoU luôn
python export_onnx.py --model checkpoints/model.pt --eval

# Giới hạn số ảnh eval
python export_onnx.py --model checkpoints/model.pt --eval --max-samples 200
```

## Visualize

```bash
python visualize.py --model checkpoints/model.pt --n 20
```

## TensorBoard

```bash
tensorboard --logdir tb_logs/
```

## Config chính (`src/config.py`)

| Param | Mặc định | Ý nghĩa |
|---|---|---|
| `pruning_ratio` | `0.5` | Tỉ lệ channel bị prune |
| `lr` | `1e-5` | Learning rate |
| `max_epochs` | `100` | Số epoch tối đa |
| `batch_size` | `4` | Batch size |
| `train_height/width` | `512/1024` | Kích thước input |

## Kết quả

| Model | Params | MACs | mIoU |
|---|---|---|---|
| EfficientViT-L2 (gốc) | ~34M | ~- G | ~-% |
| EfficientViT-L2 (pruned 50%) | ~-M | ~- G | ~-% |
