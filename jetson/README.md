
# Jetson AGX Xavier Deployment & Optimization Pipeline

This sub-directory contains the production deployment scripts to optimize, quantize, benchmark, and visualize your semantic segmentation models directly on the **NVIDIA Jetson AGX Xavier** edge platform using the **NVIDIA TensorRT** runtime engine.

---

## 🚀 Pipeline Workflow Diagram

The deployment phase shifts from PyTorch training graphs into highly parallelized, hardware-specific TensorRT engines through the following structural sequence:

```text
  [ Pruned ONNX Model ]
           │
           ├───► (FP32 Precision Build) ───────► [ FP32 Engine ]
           │
           ├───► (FP16 Precision Build) ───────► [ FP16 Engine ]
           │
           └───► [ create_calib_cache.py ] 
                        │ (Entropy Calib)
                        ▼
                 [ calib_cache.bin ]
                        │
                        ▼
                 [ export.py ] ───────────────► [ INT8 / BEST Engine ]
                                                       │
                                                       ▼
                                                [ visualize.py ]
                                           (Computes FPS & Validates mIoU)

```

---

## 📂 File Summary

* **`create_calib_cache.py`**: Generates an INT8 Entropy Calibration Cache (`calib_cache.bin`) using a representative subset of the Cityscapes dataset. This allows the compiler to scale activations accurately without clipping dynamic ranges. [Download cityscapes val dataset to generate calib_cache.bin](https://drive.google.com/drive/folders/15wYmEonL8gjfqyMSWlJ-34Wkh30d-0KS?usp=drive_link)
* **`export.py`**: An interactive terminal utility that invokes `trtexec` backend processes to compile static ONNX graphs into `.trt` engines across multiple precisions (`FP32`, `FP16`, `INT8`, `BEST`).
* **`visualize.py`**: A specialized validation and hardware-benchmarking runtime engine. It measures raw throughput (FPS), profiles localized mIoU degradation, and exports qualitative side-by-side segmentation map plots using an absolute headless rendering backend (`Agg`).

---

## 🛠️ Prerequisites & Dependencies

These scripts are built to run natively within a **Jetson AGX Xavier** environment powered by **NVIDIA JetPack (v5.x or v6.x)**.

Ensure you have the following system and Python dependencies installed on your Jetson board:

```bash
# Core System Libraries (Pre-bundled in JetPack)
# TensorRT, PyCUDA, and CUDA Toolkit must be active.

# Install required Python packages
pip install pycuda opencv-python Pillow matplotlib tqdm torch torchvision

```

*Note: Since PyTorch and TensorRT bindings on Jetson are tightly coupled to the JetPack version, ensure you use the pre-installed system `tensorrt` package.*

---

## ⚙️ Step-by-Step Deployment Guide

### Step 1: Create the INT8 Calibration Cache

To deploy using low-precision **INT8 execution paths** without losing segmentation accuracy, you must generate a quantization calibration profile using raw images (e.g., around 100-200 validation images).

Run the calibrator by targeting your model architecture and dataset path:

```bash
python create_calib_cache.py \
  --onnx /home/hieu/onnx/your_model.onnx \
  --img-dir /home/hieu/val_resized \
  --cache-file /home/hieu/calib_cache.bin \
  --num-images 200 \
  --input-h 512 \
  --input-w 1024

```

This produces a static optimization cache file at `/home/hieu/calib_cache.bin`.

### Step 2: Compile ONNX Models to TensorRT Engines

The `export.py` script automatically scans your target ONNX directory and builds hardware engines based on your selection.

1. Place your exported `.onnx` files into `/home/hieu/onnx/`.
2. Launch the deployment pipeline script:
```bash
python export.py

```


3. Interaction Prompt: The terminal will display an index of found assets. Enter specific indices (e.g., `0,2`), or type `all` to construct the entire precision array (`FP32`, `FP16`, `INT8`, `BEST`).

The completed binaries will be compiled and structured inside `/home/hieu/tensorrt/`.

### Step 3: Run Hardware Benchmarking & Visualization

To measure the final hardware performance gains (throughput vs. precision accuracy degradation), execute the validation loop script:

```bash
python visualize.py

```

**What this script handles under the hood:**

* Prompts you to pick which `.trt` engine binaries to evaluate.
* Warm-up sequences are initialized to eliminate initial GPU allocation latency.
* Measures absolute hardware execution frame rate (**FPS**).
* Iterates over the ground truth directory (`/home/hieu/val_resized_labels`) to compute the exact hardware-fused **mIoU**.
* Saves side-by-side prediction visualizations matching your configured Cityscapes color palettes directly into an output folder, using headless `matplotlib.use("Agg")` to avoid display-server dependencies on headless edge devices.


## ⚠️ Edge Troubleshooting & System Safeguards

1. **PyCUDA Context Mismatches:**
If you encounter a `pycuda._driver.LogicError: explicit_context_dependent_failed`, ensure that `cuda.init()` and `device.make_context()` calls are correctly isolated at the beginning of execution blocks and popped/detached appropriately upon thread destruction.
2. **TensorRT Memory Exhaustion (`out of memory`):**
The compilation configurations utilize up to a 4GB execution memory pool allocation (`--workspace=4096`). If your AGX Xavier is running other deep learning services or graphical desktop processes simultaneously, you may need to downscale the allocation to `--workspace=2048` or close active processes to free up RAM.
3. **Missing Calibration Cache Paths:**
The `export.py` compilation script dynamically checks for a validation token at `/home/hieu/calib_cache.bin`. If this file is missing when initiating an `INT8` or `BEST` compile routine, the execution engine flags will fall back to default dynamic scale distributions, which can severely degrade mIoU performance. Always execute **Step 1** before selecting INT8 paths.
