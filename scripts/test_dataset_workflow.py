"""End-to-end Dataset Studio test using a real licensed COCO8 photograph."""
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from dataset_studio import DatasetStudio
from training_manager import TrainingManager

ROOT = Path(__file__).resolve().parents[1]


def person_box_from_coco_polygon(label_path: Path) -> dict:
    person = next(line for line in label_path.read_text(encoding="utf-8").splitlines() if line.startswith("0 "))
    values = [float(value) for value in person.split()[1:]]
    xs, ys = values[0::2], values[1::2]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def main() -> None:
    image_path = ROOT / "datasets" / "coco8-seg" / "images" / "val" / "000000000036.jpg"
    label_path = ROOT / "datasets" / "coco8-seg" / "labels" / "val" / "000000000036.txt"
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="visionar-e2e-") as temp:
        studio = DatasetStudio(Path(temp))
        first = studio.import_candidate(image_url, image_path.name, "Ultralytics COCO8-seg bundle", "AGPL-3.0")
        duplicate = studio.import_candidate(image_url, image_path.name, "Ultralytics COCO8-seg bundle", "AGPL-3.0")
        approved = studio.approve_candidate(first["candidate"]["id"], "person", person_box_from_coco_polygon(label_path), "val")
        audit = studio.audit()
        refused_training = TrainingManager(studio).start()
        result = {"real_image": str(image_path.relative_to(ROOT)), "imported": first["imported"],
                  "duplicate_rejected": not duplicate["imported"], "approved": approved,
                  "stats": studio.stats(), "audit_ready": audit["ready"], "audit_blockers": audit["blockers"],
                  "training_refused_safely": refused_training.get("reason") == "dataset_not_ready"}
        assert result["imported"] and result["duplicate_rejected"] and approved["reviewed"]
        assert result["stats"]["reviewed_annotations"] == 1
        assert not audit["ready"], "One annotation must never unlock 38-class training"
        assert result["training_refused_safely"]
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
