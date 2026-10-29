"""
STEP 0 - Reduce the dataset's classes (e.g. 23 -> 6) by merging and dropping.

Before running:
  1. Rename your current 'dataset' folder to 'dataset_original'
  2. Run this script once with MAPPING empty -> it prints all class names
  3. Fill in MAPPING below, run again -> creates a new 'dataset' folder
Then run check_dataset.py and train.py as usual. The original data is never changed.
"""
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dataset_original"
DST = ROOT / "dataset"
SPLITS = ["train", "valid", "test"]

MAPPING = {
    "Resistor": "Resistor",
    "Resistor Network": "Resistor",
    "Resistor Jumper": "Resistor",
    "Capacitor": "Capacitor",
    "Electrolytic Capacitor": "Capacitor",
    "Capacitor Jumper": "Capacitor",
    "IC": "IC",
    "iC": "IC",
    "Connector": "Connector",
    "Pins": "Connector",
    "Transistor": "Transistor",
    "Diode": "Diode",
    "Led": "LED",
    "Button": None,
    "Clock": None,
    "EM": None,
    "Ferrite Bead": None,
    "Inductor": None,
    "Jumper": None,
    "Pads": None,
    "Switch": None,
    "Test Point": None,
    "Unknown Unlabeled": None,
}


def main():
    if not SRC.exists():
        raise SystemExit("Rename your 'dataset' folder to 'dataset_original' first.")
    with open(SRC / "data.yaml") as f:
        old_names = yaml.safe_load(f)["names"]
    if isinstance(old_names, dict):
        old_names = [old_names[k] for k in sorted(old_names)]

    if not MAPPING:
        print(f"The dataset has {len(old_names)} classes:")
        for name in old_names:
            print(f'    "{name}": ,')
        print("\nFill in MAPPING in this script, then run it again.")
        return

    mapping = {k.lower(): v for k, v in MAPPING.items()}
    new_names = list(dict.fromkeys(v for v in mapping.values() if v))
    old_to_new = {}
    for i, name in enumerate(old_names):
        target = mapping.get(name.lower())
        if target:
            old_to_new[i] = new_names.index(target)
        else:
            print(f"Dropping class: {name}")

    if DST.exists():
        raise SystemExit(f"'{DST.name}' already exists - delete it or rename it first.")

    counts = Counter()
    for split in SPLITS:
        if not (SRC / split).exists():
            continue
        (DST / split / "labels").mkdir(parents=True)
        shutil.copytree(SRC / split / "images", DST / split / "images")
        for lbl in (SRC / split / "labels").glob("*.txt"):
            out = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts and int(parts[0]) in old_to_new:
                    new_id = old_to_new[int(parts[0])]
                    out.append(" ".join([str(new_id)] + parts[1:]))
                    if split == "train":
                        counts[new_names[new_id]] += 1
            (DST / split / "labels" / lbl.name).write_text("\n".join(out))

    with open(DST / "data.yaml", "w") as f:
        yaml.safe_dump({"nc": len(new_names), "names": new_names,
                        "train": "train/images", "val": "valid/images", "test": "test/images"},
                       f, sort_keys=False)

    print(f"\nNew classes ({len(new_names)}): {new_names}")
    print("Training instances per class:")
    for name in new_names:
        print(f"    {name:<20}{counts[name]}")
    print("\nDone! Next: run check_dataset.py")


if __name__ == "__main__":
    main()