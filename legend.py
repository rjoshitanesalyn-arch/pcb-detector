"""
Creates a legend image using the EXACT colours YOLO uses for each class.
Saved to runs/legend.png
"""
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "pcb" / "weights" / "best.pt"
OUT = ROOT / "runs" / "legend.png"

model = YOLO(str(WEIGHTS))
names = model.names

# Use YOLO's own internal colour palette
colors = Colors()

# Layout settings
BOX_W      = 60
BOX_H      = 40
PADDING    = 20
TEXT_GAP   = 15
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS  = 2

n      = len(names)
img_w  = BOX_W + PADDING * 3 + 200
img_h  = PADDING * 3 + 30 + n * (BOX_H + PADDING)
img    = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

# Title
cv2.putText(img, "Component Legend", (PADDING, PADDING + 10),
            FONT, 0.8, (0, 0, 0), 2)

for i, name in names.items():
    y   = PADDING * 2 + 30 + i * (BOX_H + PADDING)
    rgb = colors(i, bgr=False)   # get exact YOLO colour for class i
    bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    # Filled colour box with border
    cv2.rectangle(img, (PADDING, y), (PADDING + BOX_W, y + BOX_H), bgr, -1)
    cv2.rectangle(img, (PADDING, y), (PADDING + BOX_W, y + BOX_H), (0, 0, 0), 1)

    # Class name
    cv2.putText(img, name,
                (PADDING + BOX_W + TEXT_GAP, y + BOX_H // 2 + 8),
                FONT, FONT_SCALE, (0, 0, 0), THICKNESS)

OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), img)
print(f"Legend saved to {OUT}")