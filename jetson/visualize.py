import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # phải đặt TRƯỚC import pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time


from PIL import Image
import os
from pathlib import Path
import torch.nn.functional as F
import torch
cuda.init()
device = cuda.Device(0)
cuda_ctx = device.make_context()


# ── 1. Load TensorRT Engine ──────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

# ── 2. Preprocess ────────────────────────────────────────────────
# Cityscapes standard normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_path, input_h=512, input_w=1024):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_w, input_h), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - MEAN) / STD
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    return img

# ── 3. Inference với TensorRT ────────────────────────────────────
def infer(engine, input_data):
    cuda_ctx.push()
    try:
        context = engine.create_execution_context()
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            name      = engine.get_tensor_name(i)
            dtype     = trt.nptype(engine.get_tensor_dtype(name))
            shape     = engine.get_tensor_shape(name)
            size      = trt.volume(shape)

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem,
                                "shape": shape})  # ← lưu shape luôn

        np.copyto(inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()

        # Trả về cả data lẫn shape thật
        return outputs[0]["host"], outputs[0]["shape"]

    finally:
        cuda_ctx.pop()  # luôn chạy dù có lỗi

# ── 4. Cityscapes label mapping ──────────────────────────────────
# 19 classes eval (ignore class 255)
CITYSCAPES_IGNORE = 255
NUM_CLASSES = 19

# Map từ trainId → evaluationId (nếu mask là raw label)
TRAIN_ID_MAP = {
    0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20,
    8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27,
    15:28, 16:31, 17:32, 18:33
}

def load_gt_mask(mask_path, input_h=512, input_w=1024):
    mask = Image.open(mask_path)
    mask = mask.resize((input_w, input_h), Image.NEAREST)
    return np.array(mask, dtype=np.int64)

# ── 5. Tính mIoU ─────────────────────────────────────────────────
def compute_iou(pred, gt, num_classes=19, ignore_index=255):
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls   = (gt == cls) & (gt != ignore_index)
        
        intersection = (pred_cls & gt_cls).sum()
        union        = (pred_cls | gt_cls).sum()
        
        if union == 0:
            continue  # class không xuất hiện → bỏ qua
        iou_list.append(intersection / union)
    
    return np.mean(iou_list)

# ── 6. Eval loop ─────────────────────────────────────────────────
def evaluate(inferencer, img_dir, mask_dir, input_h=512, input_w=1024, wasr=False):
    img_dir  = Path(img_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
    print(f"Found {len(image_paths)} images")

    conf_matrix = np.zeros((19, 19), dtype=np.int64)
    total_time  = 0.0

    for i, img_path in enumerate(image_paths):
        city      = img_path.parent.name
        stem      = img_path.stem.replace("_leftImg8bit", "")
        mask_path = mask_dir / city / f"{stem}_gtFine_labelIds.png"

        if not mask_path.exists():
            print(f"  Missing mask: {mask_path}")
            continue

        # Preprocess
        t0 = time.perf_counter()
        inp = preprocess(str(img_path), input_h, input_w)
        t1 = time.perf_counter()

        if wasr:
            inp = {"input": inp.astype(np.float32),"imu_mask": np.ones((1, 1, 1), dtype=np.float32)}

        # Inference + đo thời gian
        out, out_shape = inferencer.infer(inp)
        t2 = time.perf_counter()

        total_time += time.perf_counter() - t0

        # Postprocess
        out  = out.reshape(out_shape)
        out  = torch.from_numpy(out).float().cuda()
        out  = F.interpolate(out, size=(input_h, input_w),
                             mode="bilinear", align_corners=False)
        pred = out.argmax(dim=1).squeeze().cpu().numpy().astype(np.int64)
        t3 = time.perf_counter()

        # Load GT
        gt = np.array(
            Image.open(mask_path).resize((input_w, input_h), Image.NEAREST),
            dtype=np.int64
        )
        gt = convert_label_to_trainid(gt)
        t4 = time.perf_counter()
        if i == 0:
            print(f"preprocess : {(t1-t0)*1000:.1f} ms")
            print(f"infer      : {(t2-t1)*1000:.1f} ms")
            print(f"postprocess: {(t3-t2)*1000:.1f} ms")
            print(f"load GT    : {(t4-t3)*1000:.1f} ms")
        # Confusion matrix
        valid  = (gt != 255)
        pred_v = pred[valid]
        gt_v   = gt[valid]
        np.add.at(conf_matrix, (gt_v, pred_v), 1)

        if i % 50 == 0:
            miou_so_far, _ = compute_miou_from_confusion(conf_matrix)
            avg_ms = (total_time / (i + 1)) * 1000
            print(f"[{i:4d}/{len(image_paths)}] mIoU: {miou_so_far*100:.2f}%  |  avg infer: {avg_ms:.1f} ms")

    miou, iou_per_class = compute_miou_from_confusion(conf_matrix)
    avg_fps = len(image_paths) / total_time

    print(f"\n{'Class':<20} {'IoU':>8}")
    print("-" * 30)
    for cls, iou in enumerate(iou_per_class):
        print(f"  {CLASS_NAMES[cls]:<18} {iou*100:>7.2f}%")
    print("-" * 30)
    print(f"  {'mIoU':<18} {miou*100:>7.2f}%")
    print(f"  {'Avg infer FPS':<18} {avg_fps:.1f}")

    return miou



# ── Cityscapes 19 class colors ────────────────────────────────────
CITYSCAPES_COLORS = np.array([
    [128,  64, 128],  # 0  road
    [244,  35, 232],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [102, 102, 156],  # 3  wall
    [190, 153, 153],  # 4  fence
    [153, 153, 153],  # 5  pole
    [250, 170,  30],  # 6  traffic light
    [220, 220,   0],  # 7  traffic sign
    [107, 142,  35],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]
# Mapping labelId → trainId (Cityscapes official)
LABEL_TO_TRAIN = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7,
    21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
    28:15, 31:16, 32:17, 33:18
}

def convert_label_to_trainid(gt):
    gt_train = np.full_like(gt, 255)  # default = ignore
    for label_id, train_id in LABEL_TO_TRAIN.items():
        gt_train[gt == label_id] = train_id
    return gt_train
def compute_miou_from_confusion(conf_matrix, num_classes=19, ignore_index=255):
    iou_list = []
    for cls in range(num_classes):
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp
        union = tp + fp + fn
        if union > 0:
            iou_list.append(tp / union)
    return np.mean(iou_list), iou_list


def pred_to_color(pred_mask):
    """Chuyển mask (H, W) → ảnh màu (H, W, 3)"""
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[pred_mask == cls_id] = color
    return color_mask

def visualize(image_path, pred_mask, gt_mask=None, alpha=0.5, save_path=None):
    """
    image_path : đường dẫn ảnh gốc
    pred_mask  : numpy array (H, W) - kết quả predict
    gt_mask    : numpy array (H, W) - ground truth (optional)
    alpha      : độ trong suốt của mask overlay
    """
    # Load ảnh gốc
    img = np.array(Image.open(image_path).convert("RGB"))

    
    img = np.array(img)

    pred_color = pred_to_color(pred)

    # ── Layout ───────────────────────────────────────────────────
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    # Ảnh gốc
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Prediction overlay
    overlay = (img * (1 - alpha) + pred_color * alpha).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title("Prediction", fontsize=13)
    axes[1].axis("off")

    # Ground truth (nếu có)
    if gt_mask is not None:
        gt_color = pred_to_color(gt_mask)
        gt_overlay = (img * (1 - alpha) + gt_color * alpha).astype(np.uint8)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth", fontsize=13)
        axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=CITYSCAPES_COLORS[i] / 255, label=CLASS_NAMES[i])
        for i in range(19)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=10,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")

    plt.show()

def measure_fps(inferencer, input_h=512, input_w=1024, num_runs=100, warmup=10):
    dummy = np.random.randn(1, 3, input_h, input_w).astype(np.float32)

    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        inferencer.infer(dummy)

    print(f"Measuring FPS ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        inferencer.infer(dummy)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times) * 1000
    fps   = 1000 / np.mean(times)

    print(f"\n{'Mean latency':<20}: {np.mean(times):.2f} ms")
    print(f"{'Min latency':<20}: {np.min(times):.2f} ms")
    print(f"{'Max latency':<20}: {np.max(times):.2f} ms")
    print(f"{'Std':<20}: {np.std(times):.2f} ms")
    print(f"{'FPS':<20}: {fps:.2f}")
    return fps

class TRTInferencer:
    def __init__(self, engine):
        self.engine  = engine
        self.context = engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.inputs  = []
        self.outputs = []
        self.bindings = []

        for i in range(engine.num_io_tensors):
            name      = engine.get_tensor_name(i)
            dtype     = trt.nptype(engine.get_tensor_dtype(name))
            shape     = engine.get_tensor_shape(name)
            size      = trt.volume(shape)

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "shape": tuple(shape)
                })

    def infer(self, input_data):
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"],
                               self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"],
                               self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.outputs[0]["host"], self.outputs[0]["shape"]


# ── Chọn engine ──────────────────────────────────────────────────
ENGINE_DIR = Path("/home/hieu/tensorrt")
engine_files = sorted(ENGINE_DIR.glob("*.trt")) + sorted(ENGINE_DIR.glob("*.engine"))

if not engine_files:
    print(f"Không tìm thấy engine trong {ENGINE_DIR}")
    exit(1)

print("Danh sách TensorRT engine:")
for i, f in enumerate(engine_files):
    print(f"  [{i}] {f.name}")

print("\nNhập số thứ tự engine muốn chạy (cách nhau bằng dấu phẩy)")
print("Ví dụ: 0,2,3  hoặc  all để chọn tất cả")
choice = input(">>> ").strip()

if choice.lower() == "all":
    selected_engines = engine_files
else:
    try:
        indices = [int(x.strip()) for x in choice.split(",")]
        selected_engines = [engine_files[i] for i in indices]
    except (ValueError, IndexError):
        print("❌ Input không hợp lệ")
        exit(1)

print(f"\nSẽ chạy {len(selected_engines)} engine:")
for f in selected_engines:
    print(f"  - {f.name}")

# ── Chạy từng engine ─────────────────────────────────────────────
input_h = 512
input_w = 1024
all_results = []

for engine_path in selected_engines:
    print(f"\n{'='*55}")
    print(f"Engine: {engine_path.name}")
    print(f"{'='*55}")
    

    engine     = load_engine(str(engine_path))
    inferencer = TRTInferencer(engine)

    # Engine info
    print("=== ENGINE INFO ===")
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode  = engine.get_tensor_mode(name)
        print(f"  {'INPUT ' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'} | {name} | shape={shape} | dtype={dtype}")

    # FPS
    fps = measure_fps(inferencer, input_h=input_h, input_w=input_w)

    # mIoU
    miou = evaluate(
        inferencer = inferencer,
        img_dir    = "/home/hieu/val_resized",
        mask_dir   = "/home/hieu/val_resized_labels",
        input_h    = input_h,
        input_w    = input_w
    )

    all_results.append({
        "engine": engine_path.name,
        "fps"   : fps,
        "miou"  : miou
    })

    del inferencer
    del engine

# ── Summary tổng kết ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"{'Engine':<45} {'FPS':>8} {'mIoU':>8}")
print("-" * 65)
for r in all_results:
    print(f"  {r['engine']:<43} {r['fps']:>7.1f} {r['miou']*100:>7.2f}%")

# Cleanup CUDA
cuda_ctx.pop()
cuda_ctx.detach()
print("\nDone!")
