"""Import a licensed, bounded Open Images V7 subset into Dataset Studio.

Official human boxes become unreviewed suggestions. Human image labels only set
the intended class and always require a manually drawn box in Dataset Studio.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from dataset_studio import CLASSES, DatasetStudio
except ModuleNotFoundError:
    from scripts.dataset_studio import CLASSES, DatasetStudio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
REPORT = ROOT / "datasets" / "reports" / "openimages_visionar_import.json"
BOXES = RAW / "openimages-validation-annotations-bbox.csv"
LABELS = RAW / "oidv7-val-human-imagelabels.csv"
CLASSES_ALL = RAW / "oidv7-class-descriptions.csv"
CLASSES_BOXABLE = RAW / "oidv7-class-descriptions-boxable.csv"
METADATA = RAW / "openimages-validation-images-with-rotation.csv"

# Exact Open Images display names, reviewed against the official V7 taxonomy.
DISPLAY_NAMES = {
    "person": "Person", "eyeglasses": "Glasses", "shirt": "Shirt",
    "nail_clipper": "Nail clipper", "cell_phone": "Mobile phone",
    "perfume_bottle": "Perfume", "potted_plant": "Houseplant", "chair": "Chair",
    "marker": "Marker pen", "eraser": "Eraser", "tv": "Television", "fish": "Fish",
    "cat": "Cat", "dog": "Dog", "motorcycle": "Motorcycle", "pencil_case": "Pencil case",
    "wallet": "Wallet", "payment_card": "Payment card", "fork": "Fork", "spoon": "Spoon",
    "knife": "Knife", "pen": "Pen", "backpack": "Backpack", "handbag": "Handbag",
    "playing_card": "Playing card", "adhesive_tape": "Adhesive tape",
    "screwdriver": "Screwdriver", "paper_sheet": "Paper", "table": "Table", "sofa": "Couch",
    "cap": "Baseball cap", "sweatshirt": "Sweatshirt", "sneakers": "Sneakers",
    "wardrobe": "Wardrobe", "pillow": "Pillow", "keys": "Key",
    "usb_drive": "Usb flash drive", "banknote": "Banknote",
}


def class_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return {display: mid for mid, display in rows if mid != "LabelName"}


def load_annotations(max_images: int) -> tuple[dict, dict]:
    all_names, box_names = class_map(CLASSES_ALL), class_map(CLASSES_BOXABLE)
    missing = [display for display in DISPLAY_NAMES.values() if display not in all_names]
    if missing:
        raise RuntimeError(f"Clases ausentes en Open Images: {missing}")
    mid_to_class = {all_names[display]: canonical for canonical, display in DISPLAY_NAMES.items()}
    boxable_mids = {box_names[display] for display in DISPLAY_NAMES.values() if display in box_names}
    boxes_by_image: dict[str, list[dict]] = defaultdict(list)
    box_images_by_class: dict[str, set[str]] = defaultdict(set)
    with BOXES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = mid_to_class.get(row["LabelName"])
            if not canonical or row["LabelName"] not in boxable_mids:
                continue
            if row.get("IsDepiction") == "1" or row.get("IsGroupOf") == "1":
                continue
            xmin, xmax = float(row["XMin"]), float(row["XMax"])
            ymin, ymax = float(row["YMin"]), float(row["YMax"])
            if (xmax - xmin) * (ymax - ymin) < .0005:
                continue
            image_id = row["ImageID"]
            boxes_by_image[image_id].append({"class_name": canonical, "confidence": 1.0,
                                             "box": {"x": xmin, "y": ymin,
                                                     "w": xmax - xmin, "h": ymax - ymin}})
            box_images_by_class[canonical].add(image_id)
    label_images_by_class: dict[str, list[str]] = defaultdict(list)
    with LABELS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = mid_to_class.get(row["LabelName"])
            if canonical and float(row["Confidence"]) == 1.0:
                label_images_by_class[canonical].append(row["ImageID"])
    selected: dict[str, set[str]] = defaultdict(set)
    coverage = {}
    for canonical in CLASSES:
        box_ids = sorted(box_images_by_class.get(canonical, set()))
        label_ids = sorted(set(label_images_by_class.get(canonical, [])))
        primary = box_ids if box_ids else label_ids
        for image_id in primary[:max_images]:
            selected[image_id].add(canonical)
        if len(primary) < max_images:
            for image_id in label_ids:
                if image_id not in primary:
                    selected[image_id].add(canonical)
                    primary.append(image_id)
                    if len(primary) >= max_images:
                        break
        coverage[canonical] = {"display_name": DISPLAY_NAMES[canonical], "box_images_available": len(box_ids),
                               "human_label_images_available": len(label_ids),
                               "selected_images": min(max_images, len(primary)), "annotation_mode": "box" if box_ids else "manual"}
    return {"selected": selected, "boxes": boxes_by_image}, coverage


def load_metadata(selected_ids: set[str]) -> dict[str, dict]:
    result = {}
    with METADATA.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["ImageID"]
            if image_id in selected_ids:
                result[image_id] = row
    return result


def download_image(image_id: str) -> bytes:
    url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
    request = urllib.request.Request(url, headers={"User-Agent": "VisionAR-Dataset-Importer/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(12 * 1024 * 1024 + 1)
            if len(data) > 12 * 1024 * 1024:
                raise ValueError("imagen mayor de 12 MB")
            return data
        except Exception as exc:  # network retries are intentionally bounded
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"{image_id}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images-per-class", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_images_per_class <= 100:
        raise SystemExit("max-images-per-class debe estar entre 1 y 100")
    annotations, coverage = load_annotations(args.max_images_per_class)
    selected = annotations["selected"]
    metadata = load_metadata(set(selected))
    report = {"generated_at": datetime.now().astimezone().isoformat(),
              "source": "Open Images V7 validation",
              "official_page": "https://storage.googleapis.com/openimages/web/download_v7.html",
              "license": "CC BY 2.0 per validation metadata; verify per image before redistribution",
              "requested_per_class": args.max_images_per_class, "selected_unique_images": len(selected),
              "coverage": coverage, "dry_run": args.dry_run}
    if args.dry_run:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    studio = DatasetStudio()
    counters = Counter()
    errors = []

    def import_one(image_id: str) -> dict:
        meta = metadata.get(image_id)
        if not meta:
            raise RuntimeError(f"Falta metadata para {image_id}")
        license_url = meta.get("License", "")
        if license_url != "https://creativecommons.org/licenses/by/2.0/":
            raise RuntimeError(f"Licencia no permitida para {image_id}: {license_url}")
        raw = download_image(image_id)
        data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        source = (f"Open Images V7 validation {image_id}; landing={meta.get('OriginalLandingURL', '')}; "
                  f"author={meta.get('Author', '')}")
        imported = studio.import_candidate(data_url, f"openimages_{image_id}.jpg", source,
                                           f"CC BY 2.0 | {license_url}", reject_near=True)
        candidate_id = imported["candidate"]["id"]
        studio.set_intended_classes(candidate_id, sorted(selected[image_id]))
        suggestions = [box for box in annotations["boxes"].get(image_id, [])
                       if box["class_name"] in selected[image_id]]
        if suggestions:
            studio.set_suggestions(candidate_id, suggestions)
        return {"imported": imported["imported"], "duplicate": imported.get("duplicate"),
                "suggestions": len(suggestions)}

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as pool:
        futures = {pool.submit(import_one, image_id): image_id for image_id in sorted(selected)}
        total = len(futures)
        for index, future in enumerate(as_completed(futures), 1):
            image_id = futures[future]
            try:
                result = future.result()
                counters["imported" if result["imported"] else "duplicate"] += 1
                counters["box_suggestions"] += result["suggestions"]
            except Exception as exc:
                counters["errors"] += 1
                errors.append({"image_id": image_id, "error": str(exc)})
            if index % 25 == 0 or index == total:
                print(f"{index}/{total} | importadas={counters['imported']} duplicadas={counters['duplicate']} errores={counters['errors']}", flush=True)
    report.update({"result": dict(counters), "errors": errors[:100], "dataset_stats": studio.stats()})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": dict(counters), "stats": studio.stats(), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
