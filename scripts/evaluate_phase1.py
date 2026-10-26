"""Evaluación Fase 1: métricas COCO/Ultralytics y conteo TP/FP/FN a IoU fijo."""
from __future__ import annotations

import csv
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import psutil
import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "training/configs/phase1.yaml"
OUTPUT = ROOT / "training/metrics/phase1"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ground_truth(label: Path, width: int, height: int, target_ids: set[int]):
    items = []
    for line in label.read_text(encoding="utf-8").splitlines():
        cid, xc, yc, w, h = map(float, line.split())
        cid = int(cid)
        if cid not in target_ids: continue
        items.append((cid, ((xc-w/2)*width, (yc-h/2)*height, (xc+w/2)*width, (yc+h/2)*height)))
    return items


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_path = ROOT / cfg["model"]
    data_path = ROOT / cfg["dataset"]
    ecfg = cfg["evaluation"]
    target = {int(k): v for k, v in cfg["target_classes"].items()}
    target_ids = set(target)
    images = sorted((ROOT / "datasets/processed/phase1_coco_val/images/test").glob("*.jpg"))
    if not images: raise SystemExit("No hay imágenes Fase 1")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotated = OUTPUT / "annotated"; annotated.mkdir(exist_ok=True)
    model = YOLO(str(model_path))
    if {model.names[i] for i in target_ids} != set(target.values()):
        raise SystemExit("Las clases del peso no coinciden con la configuración")

    process = psutil.Process()
    rss_start = process.memory_info().rss
    rss_peak = rss_start
    cpu_start = sum(process.cpu_times()[:2])
    wall_start = time.perf_counter()
    times = []
    counts = {cid: Counter(tp=0, fp=0, fn=0) for cid in target_ids}
    confidences = defaultdict(list)
    error_rows = []
    prediction_rows = []
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None: raise RuntimeError(f"No se pudo leer {image_path}")
        h, w = frame.shape[:2]
        label = ROOT / "datasets/processed/phase1_coco_val/labels/test" / f"{image_path.stem}.txt"
        gt = ground_truth(label, w, h, target_ids)
        t0 = time.perf_counter()
        result = model.predict(frame, classes=sorted(target_ids), conf=ecfg["confidence"], imgsz=ecfg["imgsz"], device="cpu", verbose=False)[0]
        times.append(time.perf_counter() - t0)
        rss_peak = max(rss_peak, process.memory_info().rss)
        predictions = []
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                predictions.append((int(cls), box, float(conf)))
                confidences[int(cls)].append(float(conf))
                prediction_rows.append({"image": image_path.name, "class_id": int(cls), "class_name": target[int(cls)], "confidence": float(conf), "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        for cid in target_ids:
            gt_class = [box for c, box in gt if c == cid]
            pred_class = sorted([(box, conf) for c, box, conf in predictions if c == cid], key=lambda x: x[1], reverse=True)
            matched = set()
            for box, conf in pred_class:
                best = max(((iou(box, g), idx) for idx, g in enumerate(gt_class) if idx not in matched), default=(0.0, -1))
                if best[0] >= ecfg["iou_match"]:
                    counts[cid]["tp"] += 1; matched.add(best[1]); outcome = "TP"
                else:
                    counts[cid]["fp"] += 1; outcome = "FP"
                error_rows.append({"image": image_path.name, "class_id": cid, "class_name": target[cid], "outcome": outcome, "confidence": conf, "best_iou": best[0]})
            missed = len(gt_class) - len(matched)
            counts[cid]["fn"] += missed
            for _ in range(missed):
                error_rows.append({"image": image_path.name, "class_id": cid, "class_name": target[cid], "outcome": "FN", "confidence": "", "best_iou": ""})
        cv2.imwrite(str(annotated / image_path.name), result.plot(labels=True, conf=True))

    wall = time.perf_counter() - wall_start
    cpu_seconds = sum(process.cpu_times()[:2]) - cpu_start
    logical = psutil.cpu_count() or 1
    manual = {}
    for cid, c in counts.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        vals = confidences[cid]
        manual[str(cid)] = {"name": target[cid], **c, "precision_at_conf_0_25_iou_0_5": p, "recall_at_conf_0_25_iou_0_5": r, "detections": len(vals), "confidence_mean": sum(vals)/len(vals) if vals else None, "confidence_min": min(vals) if vals else None, "confidence_max": max(vals) if vals else None}

    metrics = model.val(data=str(data_path), split="test", classes=sorted(target_ids), device="cpu", imgsz=ecfg["imgsz"], batch=1, workers=0, plots=True, verbose=False, project=str(OUTPUT), name="ultralytics_val")
    indices = [int(x) for x in metrics.box.ap_class_index.tolist()]
    standard = {}
    for pos, cid in enumerate(indices):
        standard[str(cid)] = {"name": model.names[cid], "precision": float(metrics.box.p[pos]), "recall": float(metrics.box.r[pos]), "map50": float(metrics.box.ap50[pos]), "map50_95": float(metrics.box.ap[pos])}
    report = {
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": str(model_path.relative_to(ROOT)),
        "recognized_classes": model.names, "target_classes": target, "images": len(images),
        "settings": ecfg, "manual_at_fixed_threshold": manual,
        "ultralytics_metrics": {"overall": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}, "per_class": standard},
        "performance": {"latency_ms_mean_end_to_end": sum(times)/len(times)*1000, "latency_ms_min": min(times)*1000, "latency_ms_max": max(times)*1000, "fps_from_mean_latency": len(times)/sum(times), "wall_seconds_prediction_pass": wall, "ram_rss_start_mb": rss_start/2**20, "ram_rss_peak_mb": rss_peak/2**20, "ram_delta_mb": (rss_peak-rss_start)/2**20, "process_cpu_percent_normalized": cpu_seconds/wall/logical*100},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "opencv": cv2.__version__, "ultralytics": ultralytics_version, "cpu": platform.processor(), "logical_cpus": logical},
        "limitations": ["Subconjunto de 20 imágenes candidatas por clase, no benchmark COCO completo.", "Los conteos TP/FP/FN usan confidence=0.25 e IoU=0.50; mAP usa el barrido estándar de Ultralytics.", "Uso de CPU calculado con tiempo de proceso normalizado por núcleos lógicos; no es telemetría por núcleo."],
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for filename, rows in (("errors.csv", error_rows), ("predictions.csv", prediction_rows)):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()

