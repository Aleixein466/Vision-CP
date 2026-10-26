"""Audita y, solo con datos suficientes, entrena VisionAR Own Dataset v1."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from runtime_limits import configure_runtime
except ModuleNotFoundError:
    from scripts.runtime_limits import configure_runtime

configure_runtime()

from ultralytics import YOLO
import torch

try:
    from dataset_studio import CLASSES, DATASET_ROOT, DatasetStudio
except ModuleNotFoundError:
    from scripts.dataset_studio import CLASSES, DATASET_ROOT, DatasetStudio

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "training" / "metrics" / "visionar_own_training_readiness.json"


def audit(minimum_per_class: int) -> dict:
    studio = DatasetStudio()
    stats = studio.stats()
    records = studio._records()
    hashes_by_split = defaultdict(set)
    for record in records:
        hashes_by_split[record["sha256"]].add(record["split"])
    cross_split = sorted(digest for digest, splits in hashes_by_split.items() if len(splits) > 1)
    qualities = Counter(record["annotation"] for record in records)
    insufficient = {name: stats["by_class"][name] for name in CLASSES if stats["by_class"][name] < minimum_per_class}
    integrity = studio.audit(minimum_per_class)
    report = {
        "audited_at": datetime.now().astimezone().isoformat(), "dataset": stats,
        "minimum_per_class": minimum_per_class, "insufficient_classes": insufficient,
        "cross_split_duplicate_hashes": cross_split, "annotation_quality": dict(qualities),
        "unreviewed_annotations": sum(count for quality, count in qualities.items() if "unreviewed" in quality),
        "integrity_issues": integrity["issues"],
        "ready": not insufficient and not cross_split and not integrity["issues"] and stats["total_images"] > 0,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Entrena si la auditoría es satisfactoria")
    parser.add_argument("--minimum-per-class", type=int, default=30)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--model", default="training/checkpoints/yolo26n-objv1-seg.pt")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    report = audit(args.minimum_per_class)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    blockers = []
    if not report["ready"]:
        blockers.append("faltan imágenes por clase o existen duplicados entre splits")
    if report["unreviewed_annotations"] and not args.allow_unreviewed:
        blockers.append("hay máscaras automáticas sin revisión")
    if blockers:
        raise SystemExit("ENTRENAMIENTO_BLOQUEADO: " + "; ".join(blockers))
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(ROOT / args.model))
    results = model.train(
        data=str(DATASET_ROOT / "data.yaml"), epochs=args.epochs, imgsz=512,
        batch=2, device=device, workers=0, patience=10, cache=False,
        project=str(ROOT / "training" / "runs"), name="visionar_own_seg",
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    deploy = ROOT / "training" / "checkpoints" / "visionar_38_best.pt"
    if best.exists():
        deploy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, deploy)
    summary = {"finished_at": datetime.now().astimezone().isoformat(), "device": str(device),
               "save_dir": str(save_dir), "best_checkpoint": str(best), "deployed_checkpoint": str(deploy),
               "metrics": {key: float(value) for key, value in getattr(results, "results_dict", {}).items()
                           if isinstance(value, (int, float))}}
    (ROOT / "training" / "metrics" / "visionar_own_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
