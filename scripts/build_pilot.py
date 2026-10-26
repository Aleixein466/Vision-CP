"""Construye un piloto determinista COCO128, filtrado a cinco clases COCO."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "external" / "coco128"
OUT = ROOT / "datasets" / "processed" / "visionar_pilot_v1"
CLASSES = {1: "bicycle", 39: "bottle", 56: "chair", 63: "laptop", 67: "cell phone"}
SPLITS = (("train", 0.70), ("val", 0.20), ("test", 0.10))
SEED = 2601


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ahash(path: Path) -> int:
    with Image.open(path) as im:
        pix = list(im.convert("L").resize((8, 8)).get_flattened_data())
    mean = sum(pix) / len(pix)
    return sum((p >= mean) << i for i, p in enumerate(pix))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"No se sobrescribe el piloto: {OUT}")
    images_dir = SOURCE / "images" / "train2017"
    labels_dir = SOURCE / "labels" / "train2017"
    if not images_dir.exists():
        raise SystemExit(f"No existe {images_dir}; descargue y extraiga COCO128")

    records = []
    for image in sorted(images_dir.glob("*")):
        if not image.is_file():
            continue
        label = labels_dir / f"{image.stem}.txt"
        selected = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields and int(fields[0]) in CLASSES:
                    selected.append(line)
        if selected:
            records.append({"image": image, "label": selected, "sha256": digest(image), "ahash": ahash(image)})

    # Exactos y casi duplicados (Hamming <= 4) permanecen en el mismo grupo.
    groups: list[list[dict]] = []
    for rec in records:
        placed = False
        for group in groups:
            if rec["sha256"] == group[0]["sha256"] or (rec["ahash"] ^ group[0]["ahash"]).bit_count() <= 4:
                group.append(rec)
                placed = True
                break
        if not placed:
            groups.append([rec])
    random.Random(SEED).shuffle(groups)
    targets = {"train": round(len(records) * 0.70), "val": round(len(records) * 0.20)}
    assigned = {"train": 0, "val": 0, "test": 0}
    manifest_rows = []
    for group in groups:
        split = "train" if assigned["train"] < targets["train"] else "val" if assigned["val"] < targets["val"] else "test"
        for rec in group:
            image_dst = OUT / "images" / split / rec["image"].name
            label_dst = OUT / "labels" / split / f"{rec['image'].stem}.txt"
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec["image"], image_dst)
            label_dst.write_text("\n".join(rec["label"]) + "\n", encoding="utf-8")
            ids = sorted({int(x.split()[0]) for x in rec["label"]})
            manifest_rows.append({"file": rec["image"].name, "split": split, "sha256": rec["sha256"], "ahash": f"{rec['ahash']:016x}", "classes": ";".join(map(str, ids)), "source": "COCO128/train2017"})
            assigned[split] += 1

    manifest = ROOT / "datasets" / "manifests" / "visionar_pilot_v1.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_rows[0].keys())
        writer.writeheader(); writer.writerows(manifest_rows)
    counts = Counter()
    for row in manifest_rows:
        for class_id in row["classes"].split(";"):
            counts[CLASSES[int(class_id)]] += 1
    summary = {"seed": SEED, "selected_classes": CLASSES, "images": len(records), "split_images": assigned, "images_per_class": counts, "near_duplicate_threshold": 4}
    (ROOT / "datasets" / "reports" / "pilot_build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

