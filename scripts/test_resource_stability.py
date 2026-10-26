"""Prueba que la API descarte concurrencia sin agotar hilos o devolver 502."""
from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "datasets/processed/phase1_coco_val/images/test/000000248631.jpg"
OUTPUT = ROOT / "training/metrics/vlm/resource_stability_test.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    image = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    payload = json.dumps({
        "image": image, "confidence": 0.25, "imgsz": 320,
        "mode": "general", "scale": "n", "task": "detect",
    }).encode("utf-8")

    def detect() -> int:
        request = urllib.request.Request(
            args.base + "/api/detect", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
                return response.status
        except urllib.error.HTTPError as exc:
            exc.read()
            return exc.code

    sequential = [detect() for _ in range(20)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        concurrent = list(pool.map(lambda _: detect(), range(24)))
    report = {
        "model": "yolo26n-objv1-150.pt", "sequential": sequential,
        "concurrent_status_counts": {str(code): concurrent.count(code) for code in sorted(set(concurrent))},
        "passed": set(sequential) == {200} and set(concurrent).issubset({200, 429}) and 200 in concurrent,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("RESOURCE_STABILITY_FAILED")


if __name__ == "__main__":
    main()
