"""Prueba end-to-end del servidor web local y su proxy multimodal."""
from __future__ import annotations

import base64
import argparse
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
    with urllib.request.urlopen(f"{args.base}/", timeout=5) as response:
        html = response.read().decode("utf-8")
    with urllib.request.urlopen(f"{args.base}/api/status", timeout=5) as response:
        status = json.loads(response.read().decode("utf-8"))
    image = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    payload = json.dumps({"image": image, "prompt": "Responde solo en español: ¿qué objetos principales aparecen?"}).encode("utf-8")
    request = urllib.request.Request(f"{args.base}/api/analyze", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        analysis = json.loads(response.read().decode("utf-8"))
    report = {
        "index_ok": "VisionAR AI 2.0" in html and "Analizar escena" in html,
        "status": status,
        "analysis": analysis,
        "fusion_ok": bool(analysis.get("detections")) and "yolo" in analysis and bool(analysis.get("answer")),
        "elapsed_seconds": time.perf_counter() - start,
        "source": str(IMAGE.relative_to(ROOT)),
    }
    out = ROOT / "training/metrics/vlm/web_api_test.json"; out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["index_ok"] or not status.get("ready") or not report["fusion_ok"]:
        raise SystemExit("La validación end-to-end no cumplió los criterios")


if __name__ == "__main__": main()
