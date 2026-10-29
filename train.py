"""
STEP 2 - Train the model.

Automatically picks NVIDIA GPU / Apple GPU / CPU and sensible settings for each.
Results go to runs/pcb/ (best model = runs/pcb/weights/best.pt).
Set RESUME = True to continue a training run that was interrupted.
"""
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data_fixed.yaml"
RUN_DIR = ROOT / "runs" / "pcb"
RESUME = False


def native_image_size():
    """Largest side of a training image, so we never upscale small images."""
    img = next(p for p in (ROOT / "dataset" / "train" / "images").iterdir() if p.is_file())
    with Image.open(img) as im:
        return max(im.size)


def main():
    if not DATA.exists():
        raise SystemExit("dataset/data_fixed.yaml not found - run check_dataset.py first.")

    if RESUME:
        YOLO(str(RUN_DIR / "weights" / "last.pt")).train(resume=True)
        return

    if torch.cuda.is_available():
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        device, weights, epochs, max_size, batch = 0, "yolo11s.pt", 80, 1024, -1  # -1 = auto batch
    elif torch.backends.mps.is_available():
        print("Using Apple GPU (MPS)")
        device, weights, epochs, max_size, batch = "mps", "yolo11s.pt", 50, 640, 8
    else:
        print("WARNING: no GPU found - training on CPU will be slow (several hours).")
        device, weights, epochs, max_size, batch = "cpu", "yolo11n.pt", 30, 640, 8

    imgsz = max(320, min(max_size, native_image_size() // 32 * 32))
    print(f"Model: {weights} | image size: {imgsz} | epochs: {epochs}")

    model = YOLO(weights)  # downloads pretrained weights the first time
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        patience=20,        # stop early if no improvement for 20 epochs
        flipud=0.5,         # PCBs have no "up", so vertical flips are valid
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=True,
        plots=True,
    )
    print(f"\nDone! Best model: {RUN_DIR / 'weights' / 'best.pt'}\nNext: run evaluate.py")


if __name__ == "__main__":  # required on Windows
    main()