"""Accept only official Open Images human-verified boxes as reviewed YOLO labels."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from dataset_studio import CLASSES, DatasetStudio
except ModuleNotFoundError:
    from scripts.dataset_studio import CLASSES, DatasetStudio

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "datasets" / "reports" / "openimages_boxes_approval.json"


def split_for_sha(digest: str) -> str:
    bucket = int(digest[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket <= 2 else "train"


def main() -> None:
    studio = DatasetStudio()
    counters = Counter()
    errors = []
    for candidate in studio.list_candidates():
        if not candidate.get("source", "").startswith("Open Images V7 validation"):
            continue
        suggestions = candidate.get("suggestions", [])
        if not suggestions:
            continue
        split = split_for_sha(candidate["sha256"])
        for suggestion in suggestions:
            class_name = suggestion.get("class_name")
            if class_name not in CLASSES or float(suggestion.get("confidence", 0)) != 1.0:
                counters["skipped"] += 1
                continue
            try:
                result = studio.approve_candidate(candidate["id"], class_name, suggestion["box"], split,
                                                  "external_human_verified_rectangle")
                counters["approved"] += int(result["saved"])
                counters[class_name] += int(result["saved"])
            except Exception as exc:
                counters["errors"] += 1
                errors.append({"candidate_id": candidate["id"], "class_name": class_name, "error": str(exc)})
    report = {"approved_at": datetime.now().astimezone().isoformat(),
              "source": "Open Images V7 human xclick/activemil boxes",
              "method": "external_human_verified_rectangle", "counts": dict(counters),
              "errors": errors, "stats": studio.stats(), "audit": studio.audit()}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": dict(counters), "stats": studio.stats(), "report": str(REPORT)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
