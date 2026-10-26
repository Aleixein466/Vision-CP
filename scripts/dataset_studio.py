"""Dataset Studio: intake, human review and YOLO export for VisionAR."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from visual_fusion import decode_image_data_url
except ModuleNotFoundError:
    from scripts.visual_fusion import decode_image_data_url

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "visionar_own_v1"
CLASSES = [
    "person", "eyeglasses", "shirt", "nail_clipper", "cell_phone",
    "perfume_bottle", "potted_plant", "chair", "marker", "eraser",
    "tv", "fish", "cat", "dog", "motorcycle", "pencil_case",
    "wallet", "payment_card", "fork", "spoon", "knife", "pen",
    "backpack", "handbag", "playing_card", "adhesive_tape", "screwdriver",
    "paper_sheet", "table", "sofa", "cap", "sweatshirt", "sneakers",
    "wardrobe", "pillow", "keys", "usb_drive", "banknote",
]
SPLITS = {"train", "val", "test"}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


class DatasetStudio:
    def __init__(self, root: Path = DATASET_ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.manifest = self.root / "manifest.jsonl"
        self.candidate_manifest = self.root / "candidates.jsonl"
        self.candidate_dir = self.root / "candidates"
        self._initialize()

    def _initialize(self) -> None:
        for split in SPLITS:
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        data = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
                "names": {i: name for i, name in enumerate(CLASSES)}}
        (self.root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        metadata_path = self.root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {
            "name": "VisionAR Own Dataset v1", "created_at": _now(),
            "annotation_format": "YOLO detection/segmentation polygon",
            "provenance": "Images captured or imported explicitly by the VisionAR user",
            "license": "Per-image license/provenance in candidates.jsonl; private by default",
            "privacy": "Review captures before sharing; avoid personal or sensitive content",
        }
        metadata.update({"classes": CLASSES, "updated_at": _now()})
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    def _records(self) -> list[dict]:
        return self._read_jsonl(self.manifest)

    def _candidates(self) -> list[dict]:
        return self._read_jsonl(self.candidate_manifest)

    @staticmethod
    def _validate_box(box: dict) -> tuple[float, float, float, float]:
        try:
            x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Selecciona una caja valida antes de aprobar") from exc
        if not all(np.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("La caja contiene valores no finitos")
        if x < 0 or y < 0 or w < .01 or h < .01 or x + w > 1.0001 or y + h > 1.0001:
            raise ValueError("La caja esta fuera de la imagen o es demasiado pequena")
        return x, y, w, h

    @staticmethod
    def _canonical_jpeg(image: np.ndarray) -> bytes:
        if image is None or image.size == 0 or image.shape[0] < 16 or image.shape[1] < 16:
            raise ValueError("Imagen vacia, corrupta o demasiado pequena")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("No se pudo codificar la imagen")
        return encoded.tobytes()

    @staticmethod
    def _dhash(image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = small[:, 1:] > small[:, :-1]
        return f"{sum(int(bit) << index for index, bit in enumerate(bits.flatten())):016x}"

    @staticmethod
    def _hamming(a: str, b: str) -> int:
        return (int(a, 16) ^ int(b, 16)).bit_count()

    def import_candidate(self, image_url: str, filename: str, source: str, license_name: str,
                         reject_near: bool = False) -> dict:
        if not source.strip():
            raise ValueError("Indica la procedencia de la imagen")
        if not license_name.strip():
            raise ValueError("Indica la licencia o usa 'privada/no redistribuir'")
        image = decode_image_data_url(image_url)
        image_bytes = self._canonical_jpeg(image)
        digest = hashlib.sha256(image_bytes).hexdigest()
        perceptual = self._dhash(image)
        with self.lock:
            candidates = self._candidates()
            exact = next((r for r in candidates if r["sha256"] == digest), None)
            if exact:
                return {"imported": False, "duplicate": "exact", "candidate": exact}
            near = next((r for r in candidates if self._hamming(r.get("dhash", perceptual), perceptual) <= 3), None)
            if near and reject_near:
                return {"imported": False, "duplicate": "near", "candidate": near}
            candidate_id = uuid.uuid4().hex[:16]
            (self.candidate_dir / f"{candidate_id}.jpg").write_bytes(image_bytes)
            record = {"id": candidate_id, "filename": Path(filename or "capture.jpg").name[:180],
                      "sha256": digest, "dhash": perceptual, "width": int(image.shape[1]), "height": int(image.shape[0]),
                      "source": source.strip()[:300], "license": license_name.strip()[:200], "status": "pending",
                      "near_duplicate_of": near["id"] if near else None, "suggestions": [], "created_at": _now()}
            with self.candidate_manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"imported": True, "duplicate": "near" if near else None, "candidate": record}

    def list_candidates(self, status: str | None = None) -> list[dict]:
        records = self._candidates()
        if status:
            records = [r for r in records if r.get("status") == status]
        return sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)

    def candidate(self, candidate_id: str, include_image: bool = False) -> dict:
        record = next((r for r in self._candidates() if r["id"] == candidate_id), None)
        if not record:
            raise ValueError("Candidato no encontrado")
        result = dict(record)
        if include_image:
            raw = (self.candidate_dir / f"{candidate_id}.jpg").read_bytes()
            result["image"] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        return result

    def set_suggestions(self, candidate_id: str, detections: list[dict]) -> dict:
        clean = []
        for item in detections[:50]:
            box = item.get("box") or item.get("bbox")
            if isinstance(box, dict):
                try:
                    x, y, w, h = self._validate_box(box)
                except ValueError:
                    continue
                clean.append({"class_name": str(item.get("class_name") or item.get("class") or item.get("label", "object")),
                              "confidence": round(float(item.get("confidence", 0)), 4),
                              "box": {"x": x, "y": y, "w": w, "h": h}})
        with self.lock:
            records = self._candidates()
            record = next((r for r in records if r["id"] == candidate_id), None)
            if not record:
                raise ValueError("Candidato no encontrado")
            record.update({"suggestions": clean, "suggested_at": _now()})
            self._write_jsonl(self.candidate_manifest, records)
        return record

    def set_intended_classes(self, candidate_id: str, class_names: list[str]) -> dict:
        clean = sorted({name for name in class_names if name in CLASSES})
        with self.lock:
            records = self._candidates()
            record = next((r for r in records if r["id"] == candidate_id), None)
            if not record:
                raise ValueError("Candidato no encontrado")
            record["intended_classes"] = sorted(set(record.get("intended_classes", [])) | set(clean))
            self._write_jsonl(self.candidate_manifest, records)
        return record

    def approve_candidate(self, candidate_id: str, class_name: str, box: dict, split: str,
                          annotation_method: str = "manual_rectangle_reviewed") -> dict:
        if class_name not in CLASSES:
            raise ValueError("Clase no permitida")
        if split not in SPLITS:
            raise ValueError("Split no valido")
        if annotation_method not in {"manual_rectangle_reviewed", "external_human_verified_rectangle"}:
            raise ValueError("Metodo de anotacion no permitido")
        x, y, w, h = self._validate_box(box)
        with self.lock:
            candidates = self._candidates()
            candidate = next((r for r in candidates if r["id"] == candidate_id), None)
            if not candidate:
                raise ValueError("Candidato no encontrado")
            if candidate.get("status") == "rejected":
                raise ValueError("El candidato fue rechazado")
            image_bytes = (self.candidate_dir / f"{candidate_id}.jpg").read_bytes()
            records = self._records()
            conflicting = next((r for r in records if r["sha256"] == candidate["sha256"] and r["split"] != split), None)
            if conflicting:
                raise ValueError(f"La imagen ya pertenece al split {conflicting['split']}; no se permite fuga entre splits")
            near_conflict = next((r for r in records if r["split"] != split and r.get("dhash") and
                                  self._hamming(r["dhash"], candidate["dhash"]) <= 3), None)
            if near_conflict:
                raise ValueError(f"Existe una imagen casi identica en {near_conflict['split']}; mantenlas en el mismo split")
            stem = candidate_id
            (self.root / "images" / split / f"{stem}.jpg").write_bytes(image_bytes)
            label_path = self.root / "labels" / split / f"{stem}.txt"
            polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            row = str(CLASSES.index(class_name)) + " " + " ".join(f"{v:.6f}" for p in polygon for v in p)
            rows = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            annotation_id = hashlib.sha256(f"{candidate_id}|{class_name}|{row}".encode()).hexdigest()[:20]
            if not any(r.get("annotation_id") == annotation_id for r in records):
                rows.append(row)
                label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
                records.append({"annotation_id": annotation_id, "timestamp": _now(), "stem": stem,
                                "sha256": candidate["sha256"], "dhash": candidate["dhash"], "split": split,
                                "class_id": CLASSES.index(class_name), "class_name": class_name,
                                "source": candidate["source"], "license": candidate["license"],
                                "annotation": annotation_method, "reviewed": True,
                                "box": {"x": x, "y": y, "w": w, "h": h}, "polygon_points": 4})
                self._write_jsonl(self.manifest, records)
            candidate["status"] = "reviewed"
            candidate.setdefault("approved_annotations", [])
            if annotation_id not in candidate["approved_annotations"]:
                candidate["approved_annotations"].append(annotation_id)
            candidate["reviewed_at"] = _now()
            self._write_jsonl(self.candidate_manifest, candidates)
        return {"saved": True, "annotation_id": annotation_id, "stem": stem, "class_name": class_name,
                "split": split, "reviewed": True, "polygon": polygon}

    def reject_candidate(self, candidate_id: str) -> dict:
        with self.lock:
            records = self._candidates()
            record = next((r for r in records if r["id"] == candidate_id), None)
            if not record:
                raise ValueError("Candidato no encontrado")
            record.update({"status": "rejected", "reviewed_at": _now()})
            self._write_jsonl(self.candidate_manifest, records)
        return record

    def add_sample(self, image_url: str, class_name: str, box: dict, split: str, source: str) -> dict:
        imported = self.import_candidate(image_url, "capture.jpg", source, "privada/no redistribuir")
        return self.approve_candidate(imported["candidate"]["id"], class_name, box, split)

    def audit(self, minimum_per_class: int = 30) -> dict:
        records = self._records()
        issues: list[dict] = []
        hashes, perceptual = defaultdict(set), defaultdict(set)
        for record in records:
            hashes[record["sha256"]].add(record["split"])
            perceptual[record.get("dhash", "")].add(record["split"])
            image_path = self.root / "images" / record["split"] / f"{record['stem']}.jpg"
            label_path = self.root / "labels" / record["split"] / f"{record['stem']}.txt"
            if cv2.imread(str(image_path)) is None:
                issues.append({"type": "corrupt_or_missing_image", "path": str(image_path)})
            if not label_path.exists():
                issues.append({"type": "missing_label", "path": str(label_path)})
            if not 0 <= int(record.get("class_id", -1)) < len(CLASSES):
                issues.append({"type": "invalid_class_id", "annotation_id": record.get("annotation_id")})
            if record.get("box"):
                try:
                    self._validate_box(record["box"])
                except ValueError as exc:
                    issues.append({"type": "invalid_coordinates", "detail": str(exc)})
        for digest, splits in hashes.items():
            if len(splits) > 1:
                issues.append({"type": "exact_duplicate_across_splits", "sha256": digest, "splits": sorted(splits)})
        for digest, splits in perceptual.items():
            if digest and len(splits) > 1:
                issues.append({"type": "near_duplicate_across_splits", "dhash": digest, "splits": sorted(splits)})
        unique_images = {}
        for record in records:
            unique_images.setdefault(record["sha256"], record)
        images = list(unique_images.values())
        for index, left in enumerate(images):
            for right in images[index + 1:]:
                if left["split"] != right["split"] and left.get("dhash") and right.get("dhash") and self._hamming(left["dhash"], right["dhash"]) <= 3:
                    issue = {"type": "near_duplicate_across_splits", "left": left["stem"], "right": right["stem"],
                             "splits": sorted({left["split"], right["split"]})}
                    if issue not in issues:
                        issues.append(issue)
        stats = self.stats(minimum_per_class)
        return {"audited_at": _now(), "ready": stats["ready_for_training"] and not issues,
                "issues": issues, "blockers": len(issues), "stats": stats}

    def stats(self, minimum: int = 30) -> dict:
        records, candidates = self._records(), self._candidates()
        reviewed = [r for r in records if r.get("reviewed") is True or "unreviewed" not in r.get("annotation", "")]
        class_annotations = Counter(r["class_name"] for r in records)
        class_reviewed = Counter(r["class_name"] for r in reviewed)
        pending_suggestions = Counter()
        aliases = {"cell phone": "cell_phone", "potted plant": "potted_plant", "dining table": "table",
                   "couch": "sofa", "motorbike": "motorcycle", "hand bag": "handbag"}
        for candidate in candidates:
            if candidate.get("status") == "pending":
                suggested_classes = set()
                for suggestion in candidate.get("suggestions", []):
                    raw = suggestion.get("class_name", "").strip().lower()
                    normalized = aliases.get(raw, raw.replace(" ", "_"))
                    if normalized in CLASSES:
                        pending_suggestions[normalized] += 1
                        suggested_classes.add(normalized)
                for name in candidate.get("intended_classes", []):
                    if name in CLASSES and name not in suggested_classes:
                        pending_suggestions[name] += 1
        class_photos: dict[str, set[str]] = defaultdict(set)
        for r in records:
            class_photos[r["class_name"]].add(r["stem"])
        image_counts = {split: len(list((self.root / "images" / split).glob("*.jpg"))) for split in SPLITS}
        per_class = {name: {"photos": len(class_photos[name]), "annotations": class_annotations[name],
                            "reviewed": class_reviewed[name], "pending": class_annotations[name] - class_reviewed[name] + pending_suggestions[name],
                            "target": minimum, "missing": max(0, minimum - class_reviewed[name]),
                            "ready": class_reviewed[name] >= minimum} for name in CLASSES}
        try:
            root_display = str(self.root.relative_to(ROOT))
        except ValueError:
            root_display = str(self.root)
        return {"name": "VisionAR Own Dataset v1", "root": root_display, "classes": CLASSES,
                "images": image_counts, "total_images": sum(image_counts.values()), "annotations": len(records),
                "reviewed_annotations": len(reviewed), "pending_annotations": len(records) - len(reviewed) + sum(pending_suggestions.values()),
                "candidate_images": len(candidates), "pending_candidates": sum(r.get("status") == "pending" for r in candidates),
                "by_class": {n: class_reviewed[n] for n in CLASSES}, "per_class": per_class,
                "ready_for_training": all(class_reviewed[n] >= minimum for n in CLASSES),
                "minimum_recommended_per_class": minimum, "required_annotations": minimum * len(CLASSES)}
