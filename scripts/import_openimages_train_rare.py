"""Stream official Open Images train labels and import only rare VisionAR classes."""
from __future__ import annotations

import base64
import csv
import io
import json
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from dataset_studio import CLASSES, DatasetStudio
    from import_openimages_visionar import CLASSES_ALL, DISPLAY_NAMES, class_map
except ModuleNotFoundError:
    from scripts.dataset_studio import CLASSES, DatasetStudio
    from scripts.import_openimages_visionar import CLASSES_ALL, DISPLAY_NAMES, class_map

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "datasets" / "reports" / "openimages_train_rare_import.json"
LABEL_URL = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv"
LICENSE = "CC BY 2.0 (listed by Open Images; verify per image before redistribution)"


def select_ids(deficits: dict[str, int]) -> tuple[dict[str, set[str]], int]:
    names = class_map(CLASSES_ALL)
    mid_to_class = {names[DISPLAY_NAMES[name]]: name for name in deficits}
    selected: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    request = urllib.request.Request(LABEL_URL, headers={"User-Agent": "VisionAR-Dataset-Importer/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8", newline=""))
        for row_number, row in enumerate(reader, 1):
            canonical = mid_to_class.get(row["LabelName"])
            if canonical and row["Confidence"] == "1" and counts[canonical] < deficits[canonical]:
                image_id = row["ImageID"]
                if canonical not in selected[image_id]:
                    selected[image_id].add(canonical)
                    counts[canonical] += 1
            if row_number % 2_000_000 == 0:
                print(f"indice {row_number / 1_000_000:.0f} M filas | "
                      f"clases completas={sum(counts[n] >= deficits[n] for n in deficits)}/{len(deficits)}", flush=True)
            if all(counts[name] >= needed for name, needed in deficits.items()):
                break
    return selected, sum(counts.values())


def download(image_id: str) -> bytes:
    url = f"https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
    request = urllib.request.Request(url, headers={"User-Agent": "VisionAR-Dataset-Importer/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read(12 * 1024 * 1024 + 1)
            if len(raw) > 12 * 1024 * 1024:
                raise ValueError("imagen mayor de 12 MB")
            return raw
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(str(last_error))


def main() -> None:
    studio = DatasetStudio()
    before = studio.stats()
    deficits = {name: max(0, 30 - before["per_class"][name]["pending"] - before["per_class"][name]["reviewed"])
                for name in CLASSES}
    deficits = {name: value for name, value in deficits.items() if value}
    print("Déficits: " + json.dumps(deficits, ensure_ascii=False), flush=True)
    selected, label_count = select_ids(deficits)
    counters, errors = Counter(), []

    def import_one(image_id: str) -> dict:
        raw = download(image_id)
        data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        source = f"Open Images V7 train {image_id}; human-verified image label; {LABEL_URL}"
        result = studio.import_candidate(data_url, f"openimages_train_{image_id}.jpg", source,
                                         LICENSE, reject_near=True)
        studio.set_intended_classes(result["candidate"]["id"], sorted(selected[image_id]))
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(import_one, image_id): image_id for image_id in sorted(selected)}
        total = len(futures)
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                counters["imported" if result["imported"] else "duplicate"] += 1
            except Exception as exc:
                counters["errors"] += 1
                errors.append({"image_id": futures[future], "error": str(exc)})
            if index % 25 == 0 or index == total:
                print(f"imagenes {index}/{total} | importadas={counters['imported']} errores={counters['errors']}", flush=True)
    after = studio.stats()
    report = {"generated_at": datetime.now().astimezone().isoformat(), "source": LABEL_URL,
              "license": LICENSE, "deficits_before": deficits, "selected_unique_images": len(selected),
              "selected_human_labels": label_count, "result": dict(counters), "errors": errors,
              "stats_after": after}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": dict(counters), "pending": after["pending_annotations"],
                      "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
