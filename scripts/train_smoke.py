"""Entrenamiento corto con datos reales para validar la tubería CPU; no genera el modelo de producción."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

try:
    from runtime_limits import configure_runtime
except ModuleNotFoundError:
    from scripts.runtime_limits import configure_runtime

configure_runtime()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/checkpoints/yolo26n-seg.pt"
PROJECT = ROOT / "training/runs"
REPORT = ROOT / "training/metrics/training_smoke.json"
DATASET = ROOT / "datasets/coco8-seg"
DATA_CONFIG = ROOT / "training/configs/coco8_seg_smoke.runtime.yaml"


def main() -> None:
    started = datetime.now().astimezone()
    model = YOLO(str(MODEL))
    if not DATASET.exists():
        raise SystemExit(f"Falta el dataset real de prueba: {DATASET}")
    DATA_CONFIG.write_text(yaml.safe_dump({
        "path": DATASET.as_posix(), "train": "images/train", "val": "images/val",
        "test": None, "names": model.names,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=1,
        imgsz=320,
        batch=1,
        device="cpu",
        workers=0,
        cache=False,
        plots=False,
        val=True,
        project=str(PROJECT),
        name="cpu_training_smoke",
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights/best.pt"
    last = save_dir / "weights/last.pt"
    report = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "purpose": "CPU training pipeline validation only; not a VisionAR production checkpoint",
        "source_dataset": "COCO8-seg (real labeled subset distributed by Ultralytics)",
        "base_model": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        "settings": {"epochs": 1, "imgsz": 320, "batch": 1, "device": "cpu", "workers": 0},
        "save_dir": str(save_dir.relative_to(ROOT)).replace("\\", "/"),
        "best_checkpoint": str(best.relative_to(ROOT)).replace("\\", "/") if best.exists() else None,
        "last_checkpoint": str(last.relative_to(ROOT)).replace("\\", "/") if last.exists() else None,
        "metrics": {key: round(float(value), 6) for key, value in results.results_dict.items()},
        "production_deployed": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
