"""Valida la selección de región, YOLOE y VLM sobre una imagen real."""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "datasets/processed/phase1_coco_val/images/test/000000248631.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    image = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    payload = {
        "image": image,
        "prompt": "¿Qué objeto aparece en la zona seleccionada? No adivines.",
        "mode": "general", "confidence": 0.25, "imgsz": 320,
        "focus": {"x": 0.69, "y": 0.54, "w": 0.30, "h": 0.40},
    }
    request = urllib.request.Request(
        f"{args.base}/api/analyze", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8"))
    report = {
        "source": str(IMAGE.relative_to(ROOT)), "focus": payload["focus"],
        "focus_used": result.get("focus_used"), "mode": result.get("mode"),
        "answer": result.get("answer"), "detections": result.get("detections"),
        "yolo": result.get("yolo"), "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = ROOT / "training/metrics/vlm/focus_api_test.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["focus_used"] or report["mode"] != "general" or not report["answer"]:
        raise SystemExit("La prueba de región seleccionada no cumplió los criterios")


if __name__ == "__main__":
    main()
