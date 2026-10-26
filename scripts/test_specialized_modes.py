"""Prueba reproducible de los detectores general y abierto sin usar datos simulados."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from visual_fusion import VisualFusionEngine, crop_image_data_url, decode_image_data_url

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "datasets/processed/phase1_coco_val/images/test/000000248631.jpg"
OUTPUT = ROOT / "training/metrics/vlm/specialized_modes_test.json"


def main() -> None:
    image_url = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    engine = VisualFusionEngine()
    results = {}
    for mode, confidence, imgsz in (("general", 0.30, 480), ("small", 0.14, 640)):
        started = time.perf_counter()
        result = engine.detect(image_url, confidence, imgsz, mode)
        results[mode] = {
            "detections": result["detections"], "latency_ms": result["latency_ms"],
            "wall_seconds": round(time.perf_counter() - started, 3), "model": result["model"],
        }
    mouse = next(item for item in results["general"]["detections"] if item["class"] == "mouse")
    crop = decode_image_data_url(crop_image_data_url(image_url, mouse["box"]))
    report = {
        "source": str(IMAGE.relative_to(ROOT)), "results": results,
        "selected_crop": {"height": crop.shape[0], "width": crop.shape[1]},
        "aquarium_validation": "pending_real_user_frame",
        "note": "La imagen disponible valida selección y detector abierto; no contiene llaves, USB ni acuario.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
