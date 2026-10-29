"""
STEP 4 - Detect components in your own photos.

Put photos in the 'my_images' folder (or change SOURCE to one image file).
Annotated images are saved to runs/predictions/.
"""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "pcb" / "weights" / "best.pt"
SOURCE = ROOT / r"C:\Users\Admin\Desktop\rj\stm.jpg"  # a folder or a single image path
CONF = 0.15                  # raise to reduce false detections, lower to catch more


def main():
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found - create it and add some PCB photos.")

    model = YOLO(str(WEIGHTS))
    results = model.predict(source=str(SOURCE), conf=CONF, save=True,
                            project=str(ROOT / "runs"), name="predictions", exist_ok=True,
                            hide_conf=True, line_width=2,hide_labels=True)

    for r in results:
        counts = Counter(model.names[int(c)] for c in r.boxes.cls)
        summary = ", ".join(f"{n} x{c}" for n, c in sorted(counts.items())) or "nothing detected"
        print(f"{Path(r.path).name}: {summary}")
    print(f"\nAnnotated images saved in {ROOT / 'runs' / 'predictions'}")


if __name__ == "__main__":
    main()