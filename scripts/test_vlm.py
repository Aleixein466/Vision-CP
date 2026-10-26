"""Prueba reproducible del endpoint multimodal con una imagen real."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from vision_web_server import llama_request

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "datasets/processed/phase1_coco_val/images/test/000000248631.jpg"


def main() -> None:
    data = base64.b64encode(IMAGE.read_bytes()).decode("ascii")
    prompt = "Responde en español y en una frase: ¿qué objetos principales aparecen en la imagen? No inventes."
    start = time.perf_counter()
    result = llama_request("http://127.0.0.1:8091", f"data:image/jpeg;base64,{data}", prompt)
    result.update({"source": str(IMAGE.relative_to(ROOT)), "elapsed_seconds": time.perf_counter() - start})
    out = ROOT / "training/metrics/vlm/smoke_test.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()

