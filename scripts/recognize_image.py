"""Reconoce objetos reales en una imagen, los anota y pronuncia el resultado."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from speech_announcer import SPANISH, SpeechAnnouncer, spanish_phrase

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--no-speech", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "training/metrics/phase1/spoken_demo.jpg")
    args = p.parse_args()
    image_path = args.image.resolve()
    if not image_path.exists(): raise SystemExit(f"No existe {image_path}")
    cfg = yaml.safe_load((ROOT / "training/configs/phase1.yaml").read_text(encoding="utf-8"))
    model = YOLO(str(ROOT / cfg["model"]))
    result = model.predict(str(image_path), conf=args.confidence, imgsz=640, device="cpu", verbose=False)[0]
    ids = [int(x) for x in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
    confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    counts = Counter(model.names[x] for x in ids)
    display_names = dict(result.names)
    for cid, english in model.names.items():
        if english in SPANISH:
            display_names[cid] = SPANISH[english][0].removeprefix("un ").removeprefix("una ").removeprefix("unas ")
    result.names = display_names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result.plot(labels=True, conf=True).copy())
    phrases = [spanish_phrase(label, count) for label, count in sorted(counts.items())]
    if phrases and not args.no_speech:
        scfg = cfg["camera"]["speech"]
        announcer = SpeechAnnouncer(scfg["rate"], scfg["volume"], ROOT / "training/metrics/phase1/voice_events.jsonl")
        for phrase in phrases: announcer.say(phrase)
        announcer.flush(); announcer.close()
    report = {"source": str(image_path), "output": str(args.output.resolve()), "confidence_threshold": args.confidence, "detections": [{"class": model.names[cid], "confidence": conf} for cid, conf in zip(ids, confs)], "spoken_phrases": phrases if not args.no_speech else []}
    report_path = args.output.with_suffix(".json"); report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
