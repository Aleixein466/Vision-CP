"""Prueba HTTP de selector YOLO26, segmentación y Dataset Studio."""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "datasets/processed/phase1_coco_val/images/test/000000248631.jpg"
OUTPUT = ROOT / "training/metrics/vlm/phase2_api_test.json"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    catalog = get_json(args.base + "/api/yolo/models")
    selected = post_json(args.base + "/api/yolo/select", {"scale": "n", "task": "segment", "download": False})
    image_url = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    started = time.perf_counter()
    detection = post_json(args.base + "/api/detect", {
        "image": image_url, "confidence": 0.30, "imgsz": 320, "mode": "general",
        "scale": "n", "task": "segment",
    })
    dataset = get_json(args.base + "/api/dataset/stats")
    report = {
        "catalog_entries": len(catalog["models"]), "selected": selected["selected"],
        "detections": len(detection["detections"]),
        "masks": sum(bool(item.get("polygon")) for item in detection["detections"]),
        "latency_ms": detection["latency_ms"], "wall_seconds": round(time.perf_counter() - started, 3),
        "dataset": dataset,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["catalog_entries"] != 10 or report["masks"] < 1 or dataset["total_images"] != 0:
        raise SystemExit("La validación de Fase 2 no cumplió los criterios")


if __name__ == "__main__":
    main()
