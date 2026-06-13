# convert_all.py
import subprocess
from pathlib import Path

ONNX_DIR   = Path("/home/hieu/onnx")
ENGINE_DIR = Path("/home/hieu/tensorrt")
TRTEXEC    = "/usr/src/tensorrt/bin/trtexec"

ENGINE_DIR.mkdir(exist_ok=True)

MODES = [
    {"suffix": "fp32", "flags": []},
    {"suffix": "fp16", "flags": ["--fp16"]},
    {"suffix": "int8", "flags": ["--int8", "--fp16"]},
    {"suffix": "best", "flags": ["--best"]},  # calib được thêm tự động trong cmd
]

# ── Hiện danh sách model ─────────────────────────────────────────
onnx_files = sorted(ONNX_DIR.glob("*.onnx"))

if not onnx_files:
    print(f"Không tìm thấy file .onnx trong {ONNX_DIR}")
    exit(1)

print("Danh sách model ONNX:")
for i, f in enumerate(onnx_files):
    print(f"  [{i}] {f.name}")

# ── Chọn model ───────────────────────────────────────────────────
print("\nNhập số thứ tự model muốn convert (cách nhau bằng dấu phẩy)")
print("Ví dụ: 0,2,3  hoặc  all để chọn tất cả")
choice = input(">>> ").strip()

if choice.lower() == "all":
    selected = onnx_files
else:
    try:
        indices  = [int(x.strip()) for x in choice.split(",")]
        selected = [onnx_files[i] for i in indices]
    except (ValueError, IndexError):
        print("❌ Input không hợp lệ")
        exit(1)

# ── Chọn mode convert ────────────────────────────────────────────
print("\nChọn precision:")
print("  [0] FP32 (không quantize)")
print("  [1] FP16")
print("  [2] INT8 + FP16 fallback")
print("  [3] Best (tự chọn precision tối ưu mỗi layer)")  # ← thêm
print("  [4] FP32 + FP16")
print("  [5] FP32 + FP16 + INT8")
print("  [6] FP32 + FP16 + INT8 + Best")
print("  [7] Tất cả")
mode_choice = input(">>> ").strip()

mode_map = {
    "0": [MODES[0]],
    "1": [MODES[1]],
    "2": [MODES[2]],
    "3": [MODES[3]],                                    # ← thêm
    "4": [MODES[0], MODES[1]],
    "5": [MODES[0], MODES[1], MODES[2]],
    "6": [MODES[0], MODES[1], MODES[2], MODES[3]],     # ← thêm
    "7": MODES,                                         # ← đổi từ "5" → "7"
}
modes = mode_map.get(mode_choice, [MODES[1]])  # default FP16
if mode_choice not in mode_map:
    print("Input không hợp lệ, dùng FP16 mặc định")

# ── Xác nhận ─────────────────────────────────────────────────────
print(f"\nSẽ convert {len(selected)} model:")
for f in selected:
    print(f"  - {f.name}")
print(f"Precision: {[m['suffix'] for m in modes]}")
confirm = input("\nXác nhận? (y/n) >>> ").strip().lower()

if confirm != "y":
    print("Cancelled.")
    exit(0)

# ── Convert ──────────────────────────────────────────────────────
results = []

for onnx_file in selected:
    model_name = onnx_file.stem
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    for mode in modes:
        engine_path = ENGINE_DIR / f"{model_name}-{mode['suffix']}.trt"

        if engine_path.exists():
            print(f"  ⏭ Skip {mode['suffix']} (already exists)")
            results.append((model_name, mode['suffix'], "skipped"))
            continue

        cmd = [TRTEXEC,
            f"--onnx={onnx_file}",
            f"--saveEngine={engine_path}",
            ] + mode["flags"]

        # Thêm calib chỉ khi dùng int8 hoặc best
        if mode["suffix"] in ["int8", "best"]:
            calib_path = Path("/home/hieu/calib_cache.bin")
            if calib_path.exists():
                cmd.append(f"--calib={calib_path}")
            else:
                print(f"  ⚠ Calib cache not found, skipping --calib")

        print(f"  Converting {mode['suffix']}...")
        ret = subprocess.run(cmd, capture_output=True, text=True)

        if ret.returncode == 0:
            print(f"  ✅ {mode['suffix']} done → {engine_path.name}")
            results.append((model_name, mode['suffix'], "success"))
        else:
            print(f"  ❌ {mode['suffix']} failed")
            for line in ret.stderr.splitlines()[-5:]:
                print(f"     {line}")
            results.append((model_name, mode['suffix'], "failed"))

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for model, mode, status in results:
    icon = "✅" if status == "success" else "⏭" if status == "skipped" else "❌"
    print(f"{icon} {model:<40} {mode:<8} {status}")