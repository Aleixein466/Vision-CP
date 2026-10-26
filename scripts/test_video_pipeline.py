"""Valida lectura de video y detección continua usando fotografías reales del test COCO."""
from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path

import cv2

from visual_fusion import VisualFusionEngine

ROOT = Path(__file__).resolve().parents[1]
IMAGES = sorted((ROOT / "datasets/processed/phase1_coco_val/images/test").glob("*.jpg"))[:6]
OUTPUT = ROOT / "training/metrics/vlm/video_pipeline_test.json"


def main() -> None:
    if len(IMAGES) < 2:
        raise SystemExit("No hay suficientes imágenes reales para la prueba técnica de video")
    with tempfile.TemporaryDirectory() as temp:
        video_path = Path(temp) / "real_frames.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (640, 480))
        if not writer.isOpened():
            raise SystemExit("OpenCV no pudo crear el video temporal")
        for path in IMAGES:
            frame = cv2.imread(str(path))
            writer.write(cv2.resize(frame, (640, 480)))
        writer.release()

        capture = cv2.VideoCapture(str(video_path))
        engine = VisualFusionEngine()
        engine.warmup()
        results = []
        while len(results) < len(IMAGES):
            ok, frame = capture.read()
            if not ok:
                break
            encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not encoded_ok:
                continue
            image_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
            started = time.perf_counter()
            detection = engine.detect(image_url, 0.32, 320, "general")
            results.append({
                "frame": len(results) + 1,
                "classes": [item["class"] for item in detection["detections"]],
                "latency_ms": detection["latency_ms"],
                "wall_ms": round((time.perf_counter() - started) * 1000, 2),
            })
        capture.release()

    report = {
        "source": "temporary MP4 assembled from real COCO test photographs",
        "frames_processed": len(results), "results": results,
        "average_wall_ms": round(sum(item["wall_ms"] for item in results) / len(results), 2),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(results) != len(IMAGES):
        raise SystemExit("No se procesaron todos los fotogramas")


if __name__ == "__main__":
    main()
