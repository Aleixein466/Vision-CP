"""Audita integridad de imágenes y etiquetas YOLO sin borrar datos."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "processed" / "visionar_pilot_v1"
REPORT = ROOT / "datasets" / "reports" / "quality_report.json"
VALID_IDS = {1, 39, 56, 63, 67}


def main() -> None:
    issues = []
    hashes = defaultdict(list)
    counts = Counter()
    resolutions = Counter()
    checked = 0
    for split in ("train", "val", "test"):
        for image in sorted((DATA / "images" / split).glob("*")):
            checked += 1
            try:
                with Image.open(image) as im:
                    im.verify()
                with Image.open(image) as im:
                    resolutions[f"{im.width}x{im.height}"] += 1
                    if im.width < 64 or im.height < 64:
                        issues.append({"file": str(image), "kind": "low_resolution"})
            except Exception as exc:
                issues.append({"file": str(image), "kind": "corrupt", "detail": str(exc)})
                continue
            hashes[hashlib.sha256(image.read_bytes()).hexdigest()].append(str(image))
            label = DATA / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                issues.append({"file": str(image), "kind": "missing_label"})
                continue
            for no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    issues.append({"file": str(label), "line": no, "kind": "field_count"}); continue
                try:
                    class_id = int(fields[0]); box = list(map(float, fields[1:]))
                except ValueError:
                    issues.append({"file": str(label), "line": no, "kind": "parse_error"}); continue
                counts[class_id] += 1
                if class_id not in VALID_IDS:
                    issues.append({"file": str(label), "line": no, "kind": "unexpected_class"})
                if not all(0 <= x <= 1 for x in box) or box[2] <= 0 or box[3] <= 0:
                    issues.append({"file": str(label), "line": no, "kind": "invalid_bbox", "box": box})
                x, y, w, h = box
                if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
                    issues.append({"file": str(label), "line": no, "kind": "bbox_outside"})
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    report = {"images_checked": checked, "annotations_per_class_id": counts, "resolution_counts": resolutions, "exact_duplicate_groups": duplicates, "issue_count": len(issues), "issues": issues, "action": "No se movió ni eliminó ningún archivo; revisar antes de poblar quarantine/."}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (ROOT / "datasets" / "reports" / "quarantine_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "kind", "detail"])
        writer.writeheader()
        for issue in issues:
            writer.writerow({"file": issue.get("file"), "kind": issue.get("kind"), "detail": json.dumps(issue, ensure_ascii=False)})
    print(json.dumps({"images_checked": checked, "issue_count": len(issues), "duplicate_groups": len(duplicates)}, indent=2))


if __name__ == "__main__":
    main()

