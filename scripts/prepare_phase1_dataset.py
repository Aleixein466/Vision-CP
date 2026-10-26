"""Crea un test acotado y reproducible desde COCO val2017 oficial."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "datasets" / "external" / "annotations" / "instances_val2017.json"
RAW = ROOT / "datasets" / "external" / "phase1_coco_val2017"
OUT = ROOT / "datasets" / "processed" / "phase1_coco_val"
MANIFEST = ROOT / "datasets" / "manifests" / "phase1_coco_val.csv"
# COCO category id -> (YOLO contiguous id, name)
TARGETS = {3: (2, "car"), 44: (39, "bottle"), 73: (63, "laptop"), 77: (67, "cell phone"), 84: (73, "book")}
PER_CLASS = 20
SEED = 2602
MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not ANNOTATIONS.exists():
        raise SystemExit(f"Falta {ANNOTATIONS}")
    if MANIFEST.exists():
        raise SystemExit("El dataset Fase 1 ya existe; no se sobrescribe")
    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    images = {x["id"]: x for x in data["images"]}
    licenses = {x["id"]: x for x in data.get("licenses", [])}
    annotations_by_image = defaultdict(list)
    candidates = defaultdict(set)
    for ann in data["annotations"]:
        if ann["category_id"] in TARGETS and not ann.get("iscrowd", 0):
            annotations_by_image[ann["image_id"]].append(ann)
            candidates[ann["category_id"]].add(ann["image_id"])

    rng = random.Random(SEED)
    chosen: set[int] = set()
    selected_for = defaultdict(list)
    # Clases escasas primero; selección aleatoria determinista.
    for category_id in sorted(TARGETS, key=lambda c: len(candidates[c])):
        ids = sorted(candidates[category_id]); rng.shuffle(ids)
        preferred = [x for x in ids if x not in chosen]
        take = preferred[:PER_CLASS]
        if len(take) < PER_CLASS:
            take += [x for x in ids if x in chosen][: PER_CLASS - len(take)]
        chosen.update(take)
        selected_for[category_id] = take

    RAW.mkdir(parents=True, exist_ok=True)
    (OUT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "test").mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    rows = []
    box_counts = Counter()
    image_counts = Counter()
    for image_id in sorted(chosen):
        meta = images[image_id]
        raw_path = RAW / meta["file_name"]
        url = meta.get("coco_url") or f"http://images.cocodataset.org/val2017/{meta['file_name']}"
        if raw_path.exists():
            raise SystemExit(f"No se sobrescribe {raw_path}")
        tmp = raw_path.with_suffix(raw_path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        total_bytes += tmp.stat().st_size
        if total_bytes > MAX_DOWNLOAD_BYTES:
            raise SystemExit("Se superó el presupuesto de 150 MB")
        with Image.open(tmp) as im:
            im.verify()
        tmp.replace(raw_path)
        processed = OUT / "images" / "test" / meta["file_name"]
        shutil.copy2(raw_path, processed)
        labels = []
        present = set()
        for ann in annotations_by_image[image_id]:
            yolo_id, name = TARGETS[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / meta["width"]; yc = (y + h / 2) / meta["height"]
            wn = w / meta["width"]; hn = h / meta["height"]
            labels.append(f"{yolo_id} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
            box_counts[name] += 1; present.add(name)
        for name in present:
            image_counts[name] += 1
        (OUT / "labels" / "test" / f"{Path(meta['file_name']).stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        lic = licenses.get(meta.get("license"), {})
        rows.append({
            "image_id": image_id, "file": meta["file_name"], "source_url": url,
            "license_id": meta.get("license"), "license_name": lic.get("name"), "license_url": lic.get("url"),
            "width": meta["width"], "height": meta["height"], "sha256": sha256(raw_path),
            "classes": ";".join(sorted(present)), "source_split": "COCO val2017",
        })
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {
        "seed": SEED, "requested_images_per_class": PER_CLASS, "unique_images": len(rows),
        "images_per_class": image_counts, "boxes_per_class": box_counts, "downloaded_bytes": total_bytes,
        "annotation_source": str(ANNOTATIONS.relative_to(ROOT)), "image_source_split": "COCO val2017",
    }
    (ROOT / "datasets" / "reports" / "phase1_dataset_build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

