# create_calib_cache.py
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
from pathlib import Path
import argparse

# ── Constants ────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Calibrator ───────────────────────────────────────────────────
class CityscapesCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, img_dir, cache_file, input_h, input_w, num_images):
        super().__init__()
        self.input_h    = input_h
        self.input_w    = input_w
        self.cache_file = cache_file
        self.index      = 0

        all_imgs = sorted(Path(img_dir).rglob("*_leftImg8bit.png"))
        self.image_paths = all_imgs[:num_images]
        print(f"Found {len(self.image_paths)} calibration images")

        self.device_input = cuda.mem_alloc(
            1 * 3 * input_h * input_w * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None

        img = cv2.imread(str(self.image_paths[self.index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[None])

        cuda.memcpy_htod(self.device_input, img)
        self.index += 1

        if self.index % 50 == 0:
            print(f"  [{self.index}/{len(self.image_paths)}] calibrating...")

        return [self.device_input]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            print(f"Loading existing cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"✅ Cache saved: {self.cache_file}")


# ── Build engine để tạo cache ────────────────────────────────────
def create_calib_cache(onnx_path, img_dir, cache_file,
                       input_h, input_w, num_images):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    config.int8_calibrator = CityscapesCalibrator(
        img_dir    = img_dir,
        cache_file = cache_file,
        input_h    = input_h,
        input_w    = input_w,
        num_images = num_images
    )

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            return False

    print("Building engine (có thể mất 10-20 phút)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("❌ Build failed!")
        return False

    # Lưu engine luôn để dùng sau
    engine_path = Path(cache_file).with_suffix(".trt")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"✅ Engine saved: {engine_path}")

    return True


# ── Main ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Create TensorRT INT8 calibration cache")
    parser.add_argument("--onnx",        required=True,  help="Path to ONNX model")
    parser.add_argument("--img-dir",     required=True,  help="Path to calibration images")
    parser.add_argument("--cache-file",  default="calib_cache.bin", help="Output cache file")
    parser.add_argument("--input-h",     type=int, default=512)
    parser.add_argument("--input-w",     type=int, default=1024)
    parser.add_argument("--num-images",  type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("TensorRT INT8 Calibration Cache Creator")
    print("=" * 50)
    print(f"ONNX       : {args.onnx}")
    print(f"Image dir  : {args.img_dir}")
    print(f"Cache file : {args.cache_file}")
    print(f"Input size : {args.input_h}x{args.input_w}")
    print(f"Num images : {args.num_images}")
    print("=" * 50)

    # Init CUDA
    cuda.init()
    device   = cuda.Device(0)
    cuda_ctx = device.make_context()

    try:
        success = create_calib_cache(
            onnx_path  = args.onnx,
            img_dir    = args.img_dir,
            cache_file = args.cache_file,
            input_h    = args.input_h,
            input_w    = args.input_w,
            num_images = args.num_images
        )
        if success:
            print("\n✅ Done! Dùng cache file với trtexec:")
            print(f"   trtexec --onnx={args.onnx} \\")
            print(f"           --saveEngine=model_int8.trt \\")
            print(f"           --int8 --fp16 \\")
            print(f"           --calib={args.cache_file}")
    finally:
        cuda_ctx.pop()
        cuda_ctx.detach()

