"""Detector multiclase en cámara para CPU, con controles interactivos."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from speech_announcer import SPANISH, SpeechAnnouncer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training/configs/phase1.yaml"


def open_camera(index: int, width: int, height: int):
    backend = cv2.CAP_DSHOW if __import__("os").name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cameras(max_index: int, width=640, height=480):
    found = []
    for index in range(max_index + 1):
        cap = open_camera(index, width, height)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok and frame is not None:
            found.append({"index": index, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        cap.release()
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="VisionAR AI - detección por cámara")
    p.add_argument("--camera", type=int)
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--smoke-frames", type=int, default=0, help="Lee e infiere N frames sin abrir ventana")
    p.add_argument("--all-coco", action="store_true", help="Detecta las 80 clases; por defecto solo las cinco de Fase 1")
    p.add_argument("--no-speech", action="store_true", help="Desactiva la salida de voz")
    p.add_argument("--test-speech", action="store_true", help="Pronuncia una frase de prueba y termina")
    args = p.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); ccfg = cfg["camera"]
    scfg = ccfg["speech"]
    if args.test_speech:
        announcer = SpeechAnnouncer(scfg["rate"], scfg["volume"], ROOT / "training/metrics/phase1/voice_events.jsonl")
        announcer.say("VisionAR AI está listo para reconocer objetos"); announcer.flush(); announcer.close()
        print("Prueba de voz completada"); return
    available = list_cameras(ccfg["max_camera_index"])
    if args.list_cameras:
        print(available)
        if args.camera is None and not args.smoke_frames: return
    if not available: raise SystemExit("No se encontró una cámara accesible")
    if args.camera is None and len(available) > 1 and sys.stdin.isatty():
        print("Cámaras disponibles:", available)
        index = int(input("Seleccione índice de cámara: ").strip())
    else:
        index = args.camera if args.camera is not None else available[0]["index"]
    if index not in {x["index"] for x in available}: raise SystemExit(f"Cámara {index} no disponible: {available}")
    width, height = ccfg["resolution"]
    cap = open_camera(index, width, height)
    model = YOLO(str(ROOT / cfg["model"]))
    classes = None if args.all_coco else sorted(int(x) for x in cfg["target_classes"])
    confidence = float(ccfg["confidence"]); sizes = [320, 480, 640, 800]; size_pos = sizes.index(ccfg["imgsz"]) if ccfg["imgsz"] in sizes else 1
    labels = bool(ccfg["show_labels"]); frame_no = 0; processed = 0; last = time.perf_counter()
    speech_enabled = bool(scfg["enabled"]) and not args.no_speech
    announcer = SpeechAnnouncer(scfg["rate"], scfg["volume"], ROOT / "training/metrics/phase1/voice_events.jsonl") if speech_enabled else None
    visible_streak = defaultdict(int); absent_streak = defaultdict(int); active_spoken = set(); last_spoken = defaultdict(lambda: float("-inf"))
    try:
        while True:
            ok, frame = cap.read()
            if not ok: raise RuntimeError("La cámara dejó de entregar frames")
            frame_no += 1
            if frame_no % (int(ccfg["frame_skip"]) + 1) != 0: continue
            result = model.predict(frame, classes=classes, conf=confidence, imgsz=sizes[size_pos], device="cpu", verbose=False)[0]
            processed += 1
            ids = [int(x) for x in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
            confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
            counts = Counter(model.names[x] for x in ids)
            strong = Counter(model.names[cid] for cid, conf in zip(ids, confs) if conf >= scfg["min_confidence"])
            now = time.perf_counter()
            for label in model.names.values():
                if label in strong:
                    visible_streak[label] += 1; absent_streak[label] = 0
                    if speech_enabled and visible_streak[label] >= scfg["stable_frames"] and label not in active_spoken and now - last_spoken[label] >= scfg["cooldown_seconds"]:
                        announcer.say_object(label, strong[label]); active_spoken.add(label); last_spoken[label] = now
                else:
                    visible_streak[label] = 0; absent_streak[label] += 1
                    if absent_streak[label] >= scfg["absent_frames_to_reset"]: active_spoken.discard(label)
            display_names = dict(result.names)
            for class_id, english in model.names.items():
                if english in SPANISH:
                    display_names[class_id] = SPANISH[english][0].removeprefix("un ").removeprefix("una ")
            result.names = display_names
            view = result.plot(labels=labels, conf=labels).copy()
            fps = 1 / max(now - last, 1e-9); last = now
            status = f"cam={index} conf={confidence:.2f} imgsz={sizes[size_pos]} FPS={fps:.1f} labels={'on' if labels else 'off'} voz={'on' if speech_enabled else 'off'}"
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            count_text = []
            for label, count in sorted(counts.items()):
                singular, plural = SPANISH.get(label, (label, label))
                noun = singular.removeprefix("un ").removeprefix("una ") if count == 1 else plural
                count_text.append(f"{noun}={count}")
            cv2.putText(view, "Conteo: " + ", ".join(count_text), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if args.smoke_frames and processed >= args.smoke_frames:
                smoke = {"camera": index, "frames_inferred": processed, "last_counts": dict(counts), "last_fps": fps, "confidence": confidence, "imgsz": sizes[size_pos], "source": "live_camera"}
                smoke_dir = ROOT / "training/metrics/phase1"; smoke_dir.mkdir(parents=True, exist_ok=True)
                (smoke_dir / "camera_smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
                cv2.imwrite(str(smoke_dir / "camera_smoke_last.jpg"), view)
                print(smoke); break
            cv2.imshow("VisionAR AI - q/ESC salir | l etiquetas | s voz | +/- confianza | [/ ] resolucion", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")): break
            if key == ord("l"): labels = not labels
            elif key == ord("s") and announcer is not None: speech_enabled = not speech_enabled
            elif key in (ord("+"), ord("=")): confidence = min(0.95, confidence + 0.05)
            elif key in (ord("-"), ord("_")): confidence = max(0.05, confidence - 0.05)
            elif key == ord("["): size_pos = max(0, size_pos - 1)
            elif key == ord("]"): size_pos = min(len(sizes) - 1, size_pos + 1)
    finally:
        cap.release(); cv2.destroyAllWindows()
        if announcer is not None: announcer.close()


if __name__ == "__main__": main()
