"""Evalúa YOLO26n en CPU y guarda métricas estructuradas."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import psutil
from ultralytics import YOLO, __version__

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training" / "checkpoints" / "yolo26n.pt"
CLASSES = [1, 39, 56, 63, 67]


def run(data: Path, split: str, name: str) -> dict:
    model = YOLO(str(MODEL))
    start = time.perf_counter()
    metrics = model.val(
        data=str(data), split=split, classes=CLASSES, device="cpu", imgsz=640,
        batch=1, workers=0, plots=True, verbose=False,
        project=str(ROOT / "training" / "metrics" / "runs"), name=name,
    )
    elapsed = time.perf_counter() - start
    speed = {k: float(v) for k, v in metrics.speed.items()}
    per_class = {}
    present = getattr(metrics.box, "ap_class_index", [])
    maps = getattr(metrics.box, "maps", [])
    for class_id in CLASSES:
        per_class[str(class_id)] = {
            "name": model.names[class_id],
            "map50_95": float(maps[class_id]) if len(maps) > class_id and class_id in present else None,
        }
    return {
        "split": split, "data": str(data.relative_to(ROOT)), "classes": CLASSES,
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "per_class": per_class, "speed_ms_per_image": speed,
        "elapsed_seconds_including_setup": elapsed,
    }


def main() -> None:
    report = {
        "model": "yolo26n.pt",
        "model_sha256": __import__("hashlib").sha256(MODEL.read_bytes()).hexdigest(),
        "ultralytics": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "cpu_logical_count": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "settings": {"device": "cpu", "imgsz": 640, "batch": 1, "workers": 0},
        "held_out_test": run(ROOT / "CONFIGURACION_YOLO.yaml", "test", "held_out_test"),
        "diagnostic_all_21": run(ROOT / "training" / "configs" / "pilot_all_eval.yaml", "val", "diagnostic_all_21"),
        "warning": "Piloto diminuto y derivado de COCO train2017, dominio cercano al preentrenamiento. Métricas exploratorias; no estiman producción/AR.",
    }
    out = ROOT / "training" / "metrics" / "baseline_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
