"""Auditoría de integridad, formato y fuga entre splits/datasets."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    "pilot_train": ROOT / "datasets/processed/visionar_pilot_v1/images/train",
    "pilot_val": ROOT / "datasets/processed/visionar_pilot_v1/images/val",
    "pilot_test": ROOT / "datasets/processed/visionar_pilot_v1/images/test",
    "phase1_test": ROOT / "datasets/processed/phase1_coco_val/images/test",
}
EXPECTED = {2, 39, 63, 67, 73}


def ahash(path: Path) -> int:
    with Image.open(path) as im:
        px = list(im.convert("L").resize((8, 8)).get_flattened_data())
    avg = sum(px) / len(px)
    return sum((value >= avg) << i for i, value in enumerate(px))


def main() -> None:
    issues = []; records = []; class_boxes = Counter()
    for split, folder in SETS.items():
        if not folder.exists():
            issues.append({"kind": "missing_directory", "path": str(folder)}); continue
        for image in sorted(folder.glob("*")):
            if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            try:
                with Image.open(image) as im: im.verify()
            except Exception as exc:
                issues.append({"kind": "corrupt_image", "file": str(image), "detail": str(exc)}); continue
            label = Path(str(image).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
            if not label.exists():
                issues.append({"kind": "missing_label", "file": str(image)}); continue
            for no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    issues.append({"kind": "field_count", "file": str(label), "line": no}); continue
                try: cid = int(fields[0]); vals = list(map(float, fields[1:]))
                except ValueError:
                    issues.append({"kind": "parse", "file": str(label), "line": no}); continue
                if split == "phase1_test" and cid not in EXPECTED:
                    issues.append({"kind": "unexpected_class", "file": str(label), "line": no, "class_id": cid})
                if not all(0 <= x <= 1 for x in vals) or vals[2] <= 0 or vals[3] <= 0:
                    issues.append({"kind": "invalid_bbox", "file": str(label), "line": no, "values": vals})
                class_boxes[(split, cid)] += 1
            raw = image.read_bytes()
            records.append({"split": split, "file": str(image), "sha256": hashlib.sha256(raw).hexdigest(), "ahash": ahash(image)})
    exact = defaultdict(list)
    for rec in records: exact[rec["sha256"]].append(rec)
    exact_cross = [v for v in exact.values() if len({x["split"] for x in v}) > 1]
    near_cross = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a["split"] != b["split"] and (a["ahash"] ^ b["ahash"]).bit_count() <= 4:
                near_cross.append({"a": a["file"], "b": b["file"], "distance": (a["ahash"] ^ b["ahash"]).bit_count()})
    manifest = ROOT / "datasets/manifests/phase1_coco_val.csv"
    provenance_rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline=""))) if manifest.exists() else []
    missing_provenance = [r["file"] for r in records if r["split"] == "phase1_test" and not any(x["file"] == Path(r["file"]).name for x in provenance_rows)]
    report = {
        "images_checked": len(records), "images_by_split": Counter(x["split"] for x in records),
        "phase1_boxes_by_class": {str(cid): count for (split, cid), count in class_boxes.items() if split == "phase1_test"},
        "issue_count": len(issues), "issues": issues, "exact_cross_split_duplicates": exact_cross,
        "near_cross_split_duplicates_hamming_le_4": near_cross, "missing_provenance": missing_provenance,
        "license_rows_with_name_and_url": sum(bool(x.get("license_name") and x.get("license_url")) for x in provenance_rows),
    }
    out = ROOT / "datasets/reports/phase1_audit.json"; out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"images_checked": len(records), "issues": len(issues), "exact_cross": len(exact_cross), "near_cross": len(near_cross), "missing_provenance": len(missing_provenance)}, indent=2))


if __name__ == "__main__": main()
