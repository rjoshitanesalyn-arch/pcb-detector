"""
STEP 1 - Check the dataset.

- Prints the class names and how many labelled components each split has
- Writes dataset/data_fixed.yaml with absolute paths
  (Roboflow's data.yaml often uses '../train/images' paths that break locally)
"""
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(r"C:\Users\Admin\PycharmProjects\PythonProject\dataset")
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # yaml key -> folder name
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"Could not find {yaml_path}\n"
                         "Unzip the Roboflow download into a folder named 'dataset' next to this script.")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    print(f"Classes ({len(names)}): {names}\n")

    fixed = {"path": str(DATASET_DIR), "nc": len(names), "names": names}

    for key, folder in SPLITS.items():
        img_dir = DATASET_DIR / folder / "images"
        lbl_dir = DATASET_DIR / folder / "labels"
        if not img_dir.exists():
            print(f"[{folder}] not found - skipped\n")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        counts, unlabelled = Counter(), 0
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                unlabelled += 1
                continue
            for line in lbl.read_text().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1

        fixed[key] = f"{folder}/images"
        print(f"[{folder}] {len(images)} images ({unlabelled} without labels)")
        for i, name in enumerate(names):
            n = counts.get(i, 0)
            warn = "   <-- very few, this class may be weak" if key == "train" and n < 50 else ""
            print(f"    {name:<22}{n:>6}{warn}")

        if images:
            with Image.open(images[0]) as im:
                print(f"    sample image size: {im.size[0]} x {im.size[1]}")
        print()

    if "train" not in fixed or "val" not in fixed:
        raise SystemExit("Need both 'train' and 'valid' folders inside dataset/.")

    out = DATASET_DIR / "data_fixed.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(fixed, f, sort_keys=False)
    print(f"OK - wrote {out}\nNext: run train.py")


if __name__ == "__main__":
    main()