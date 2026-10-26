"""Batch-accept operator-reviewed Open Images boxes without weakening auditability.

The command requires --confirm-reviewed. It only accepts human-verified Open
Images boxes with traceable CC-BY provenance; it never approves model guesses
or photographs that have no verified bounding box.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from dataset_studio import CLASSES, DatasetStudio
except ModuleNotFoundError:
    from scripts.dataset_studio import CLASSES, DatasetStudio


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "datasets" / "reports" / "openimages_human_batch_acceptance.json"
METHOD = "batch_human_confirmed_openimages_rectangle"


def split_for_sha(digest: str) -> str:
    """Stable 80/20/10-ish image-level split."""
    bucket = int(digest[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket <= 2 else "train"


def eligible(candidate: dict) -> bool:
    suggestions = candidate.get("suggestions", [])
    return (
        candidate.get("status") == "pending"
        and candidate.get("source", "").startswith("Open Images V7")
        and candidate.get("provenance_traceable") is True
        and bool(suggestions)
        and all(
            item.get("class_name") in CLASSES
            and float(item.get("confidence", 0)) == 1.0
            and isinstance(item.get("box"), dict)
            for item in suggestions
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-reviewed", action="store_true",
                        help="record the operator's explicit visual acceptance")
    args = parser.parse_args()
    if not args.confirm_reviewed:
        parser.error("Refusing to count labels as reviewed without --confirm-reviewed")

    studio = DatasetStudio()
    reconciliation = studio.reconcile()
    candidates = [item for item in studio.list_candidates() if eligible(item)]
    backup = studio.store.backup("before-openimages-human-batch-acceptance")
    counters, errors = Counter(), []

    for candidate in candidates:
        split = split_for_sha(candidate["sha256"])
        saved = 0
        for suggestion in candidate["suggestions"]:
            try:
                studio.approve_candidate(candidate["id"], suggestion["class_name"], suggestion["box"],
                                         split, METHOD)
                counters["annotations_accepted"] += 1
                counters[f"class:{suggestion['class_name']}"] += 1
                saved += 1
            except (OSError, ValueError) as exc:
                errors.append({"candidate_id": candidate["id"], "class_name": suggestion.get("class_name"),
                               "error": str(exc)})
        if saved == len(candidate["suggestions"]):
            current = studio.candidate(candidate["id"])
            current.pop("annotations", None)
            current["review_complete"] = True
            current["review_scope"] = "operator-confirmed Open Images boxes"
            current["review_method"] = METHOD
            current["reviewed_at"] = datetime.now().astimezone().isoformat()
            studio.store.put("candidate", [current])
            counters["images_accepted"] += 1

    report = {
        "accepted_at": datetime.now().astimezone().isoformat(),
        "operator_confirmation": "--confirm-reviewed",
        "method": METHOD,
        "selection": "Pending Open Images V7 candidates with traceable provenance and confidence=1.0 boxes",
        "backup": str(backup),
        "reconciliation_before": reconciliation,
        "counts_added_this_run": dict(counters),
        "accepted_totals": {
            "images": sum(
                candidate.get("review_method") == METHOD
                for candidate in studio.list_candidates()
            ),
            "annotations": sum(
                record.get("annotation") == METHOD and not record.get("withdrawn_at")
                for record in studio._records()
            ),
        },
        "errors": errors,
        "stats_after": studio.stats(),
        "audit_after": studio.audit(),
        "not_accepted": {
            "reason": "No verified bounding box or source outside the explicitly accepted Open Images batch",
            "remaining_pending_images": sum(c.get("status") == "pending" for c in studio.list_candidates()),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": dict(counters), "errors": errors, "report": str(REPORT),
                      "audit_ready": report["audit_after"]["ready"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
