"""Dataset Studio: intake, human review and YOLO export for VisionAR."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import uuid
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from dataset_store import DatasetStore, transactional
except ModuleNotFoundError:
    from scripts.dataset_store import DatasetStore, transactional

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
CLASS_ALIASES = {
    **{name: name for name in CLASSES},
    "cell phone": "cell_phone", "mobile phone": "cell_phone", "potted plant": "potted_plant",
    "dining table": "table", "couch": "sofa", "tvmonitor": "tv", "tv monitor": "tv",
    "purse": "handbag", "glasses": "eyeglasses", "spectacles": "eyeglasses",
    "sneaker": "sneakers", "shoe": "sneakers", "usb flash drive": "usb_drive",
    "flash drive": "usb_drive", "marker pen": "marker",
}
REVIEWED_ANNOTATION_METHODS = {
    "manual_rectangle_reviewed",
    "manual_polygon_reviewed",
    "assisted_polygon_reviewed",
    # Created only by the explicit-confirmation batch command. This prevents
    # model suggestions from silently becoming human-reviewed labels.
    "batch_human_confirmed_openimages_rectangle",
    "capture_user_confirmed_yolo_rectangle",
}


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
        self.store = DatasetStore(self.root)

    def _initialize(self) -> None:
        for split in SPLITS:
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        data = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
                "names": {i: name for i, name in enumerate(CLASSES)}}
        if not (self.root / 'data.yaml').exists():
            (self.root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        metadata_path = self.root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {
            "name": "VisionAR Own Dataset v1", "created_at": _now(),
            "annotation_format": "YOLO detection/segmentation polygon",
            "provenance": "Images captured or imported explicitly by the VisionAR user",
            "license": "Per-image license/provenance in candidates.jsonl; private by default",
            "privacy": "Review captures before sharing; avoid personal or sensitive content",
        }
        if not metadata_path.exists():
            metadata.update({"classes": CLASSES, "updated_at": _now()})
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        # Compatibility for importers. SQLite is authoritative after migration.
        self.store.put('candidate' if path == self.candidate_manifest else 'annotation', records)

    def _records(self) -> list[dict]:
        return self.store.records('annotation')

    def _candidates(self) -> list[dict]:
        return self.store.records('candidate')

    def _candidate_path(self, candidate_id):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', candidate_id):
            raise ValueError('Identificador de imagen no válido')
        return self.candidate_dir / f'{candidate_id}.jpg'

    @staticmethod
    def is_reviewed(record):
        return (not record.get('withdrawn_at') and record.get('reviewed') is True
                and record.get('annotation') in REVIEWED_ANNOTATION_METHODS)

    @staticmethod
    def _validate_box(box: dict) -> tuple[float, float, float, float]:
        try:
            x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Selecciona una caja valida antes de aprobar") from exc
        if not all(np.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("La caja contiene valores no finitos")
        if x < 0 or y < 0 or w < .0001 or h < .0001 or x + w > 1.0001 or y + h > 1.0001:
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

    @staticmethod
    def normalize_class_name(value: str) -> str | None:
        """Map detector vocabulary to the stable internal YOLO class identifier."""
        key = str(value or "").strip().lower().replace("-", " ")
        return CLASS_ALIASES.get(key) or CLASS_ALIASES.get(key.replace(" ", "_"))

    @transactional
    def import_candidate(self, image_url: str, filename: str, source: str, license_name: str,
                         reject_near: bool = False, group_id: str = '') -> dict:
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
            near = next((r for r in candidates if r.get('dhash') and self._hamming(r['dhash'], perceptual) <= 3), None)
            if near and reject_near:
                return {"imported": False, "duplicate": "near", "candidate": near}
            candidate_id = uuid.uuid4().hex[:16]
            (self.candidate_dir / f"{candidate_id}.jpg").write_bytes(image_bytes)
            record = {"id": candidate_id, "filename": Path(filename or "capture.jpg").name[:180],
                      "sha256": digest, "dhash": perceptual, "width": int(image.shape[1]), "height": int(image.shape[0]),
                      "source": source.strip()[:300], "license": license_name.strip()[:200], "status": "pending",
                      "near_duplicate_of": near["id"] if near else None, "suggestions": [], "created_at": _now()}
            record['private'] = True
            record['group_id'] = str(group_id).strip()[:120] or None
            self.store.put('candidate', [record])
        return {"imported": True, "duplicate": "near" if near else None, "candidate": record}

    def list_candidates(self, status: str | None = None) -> list[dict]:
        records = self._candidates()
        if status:
            records = [r for r in records if r.get("status") == status]
        return sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)

    def candidate(self, candidate_id: str, include_image: bool = False) -> dict:
        path = self._candidate_path(candidate_id)
        record = next((r for r in self._candidates() if r["id"] == candidate_id), None)
        if not record:
            raise ValueError("Candidato no encontrado")
        result = dict(record)
        result['annotations']=[r for r in self._records() if r.get('stem')==candidate_id]
        if include_image:
            raw = path.read_bytes()
            result["image"] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        return result

    @transactional
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

    @transactional
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

    @transactional
    def approve_candidate(self, candidate_id: str, class_name: str, box: dict, split: str,
                          annotation_method: str = "manual_rectangle_reviewed") -> dict:
        if class_name not in CLASSES:
            raise ValueError("Clase no permitida")
        if split not in SPLITS:
            raise ValueError("Split no valido")
        if annotation_method not in {
            'manual_rectangle_reviewed', 'batch_human_confirmed_openimages_rectangle',
            'capture_user_confirmed_yolo_rectangle',
        }:
            raise ValueError('Las etiquetas externas requieren revisión humana explícita en Dataset Studio')
        x, y, w, h = self._validate_box(box)
        with self.lock:
            candidates = self._candidates()
            candidate = next((r for r in candidates if r["id"] == candidate_id), None)
            if not candidate:
                raise ValueError("Candidato no encontrado")
            if candidate.get("status") == "rejected":
                raise ValueError("El candidato fue rechazado")
            image_bytes = self._candidate_path(candidate_id).read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != candidate['sha256']:
                raise ValueError('La imagen cambió desde su importación; reconciliar antes de revisar')
            records = self._records()
            if candidate.get('group_id'):
                members={c['id'] for c in candidates if c.get('group_id')==candidate['group_id']}
                if any(r['stem'] in members and r['split']!=split for r in records):
                    raise ValueError('La sesión de captura ya pertenece a otro split')
            if candidate.get('duplicate_group'):
                members={c['id'] for c in candidates if c.get('duplicate_group')==candidate['duplicate_group']}
                if any(r['stem'] in members and r['split']!=split for r in records):
                    raise ValueError('El grupo de duplicados debe conservar el mismo split')
            conflicting = next((r for r in records if r["sha256"] == candidate["sha256"] and r["split"] != split), None)
            if conflicting:
                raise ValueError(f"La imagen ya pertenece al split {conflicting['split']}; no se permite fuga entre splits")
            near_conflict = next((r for r in records if r["split"] != split and r.get("dhash") and
                                  self._hamming(r["dhash"], candidate["dhash"]) <= 3), None)
            if near_conflict:
                raise ValueError(f"Existe una imagen casi identica en {near_conflict['split']}; mantenlas en el mismo split")
            stem = candidate_id
            row = str(CLASSES.index(class_name)) + ' ' + ' '.join(f'{v:.6f}' for v in (x+w/2, y+h/2, w, h))
            annotation_id = hashlib.sha256(f"{candidate_id}|{class_name}|{row}".encode()).hexdigest()[:20]
            existing=next((r for r in records if r.get('annotation_id')==annotation_id),None)
            if existing and existing.get('withdrawn_at'):
                annotation_id=hashlib.sha256(f'{annotation_id}|{_now()}'.encode()).hexdigest()[:20]
            if not any(r.get("annotation_id") == annotation_id for r in records):
                records.append({"annotation_id": annotation_id, "timestamp": _now(), "stem": stem,
                                "sha256": candidate["sha256"], "dhash": candidate["dhash"], "split": split,
                                "class_id": CLASSES.index(class_name), "class_name": class_name,
                                "source": candidate["source"], "license": candidate["license"],
                                "annotation": annotation_method, "reviewed": True,
                                "box": {"x": x, "y": y, "w": w, "h": h}, "task": "detect"})
                self._write_jsonl(self.manifest, records)
            candidate["status"] = "reviewed"
            candidate.setdefault("approved_annotations", [])
            if annotation_id not in candidate["approved_annotations"]:
                candidate["approved_annotations"].append(annotation_id)
            candidate["reviewed_at"] = _now()
            self._write_jsonl(self.candidate_manifest, candidates)
        return {"saved": True, "annotation_id": annotation_id, "stem": stem, "class_name": class_name,
                "split": split, "reviewed": True, "task": "detect", "polygon": None}

    @transactional
    def approve_camera_detections(self, candidate_id: str, detections: list[dict], split: str) -> dict:
        """Persist a user's explicit acceptance of YOLO boxes from one camera frame."""
        if not isinstance(detections, list) or len(detections) > 100:
            raise ValueError("Cantidad de detecciones no válida")
        approved, skipped = [], []
        for detection in detections:
            raw_class = detection.get("class_name") or detection.get("class") or detection.get("label")
            class_name = self.normalize_class_name(raw_class)
            if not class_name:
                skipped.append(str(raw_class or "sin clase"))
                continue
            approved.append(self.approve_candidate(candidate_id, class_name, detection.get("box"), split,
                                                   "capture_user_confirmed_yolo_rectangle"))
        candidate = self.candidate(candidate_id)
        candidate.pop("annotations", None)
        candidate["review_complete"] = bool(approved)
        candidate["review_scope"] = "user-confirmed live camera detections"
        candidate["review_method"] = "capture_user_confirmed_yolo_rectangle"
        candidate["reviewed_at"] = _now()
        self.store.put("candidate", [candidate])
        return {"candidate_id": candidate_id, "approved": approved, "skipped": skipped,
                "review_complete": bool(approved)}

    @transactional
    def reject_candidate(self, candidate_id: str) -> dict:
        with self.lock:
            records = self._candidates()
            record = next((r for r in records if r["id"] == candidate_id), None)
            if not record:
                raise ValueError("Candidato no encontrado")
            if any(r.get('stem') == candidate_id and self.is_reviewed(r) for r in self._records()):
                raise ValueError('La imagen tiene anotaciones aprobadas; no se puede rechazar sin revisarlas')
            record.update({"status": "rejected", "reviewed_at": _now()})
            self._write_jsonl(self.candidate_manifest, records)
        return record

    @transactional
    def withdraw_annotation(self, annotation_id):
        record=next((r for r in self._records() if r.get('annotation_id')==annotation_id),None)
        if not record:
            raise ValueError('Anotación no encontrada')
        record['withdrawn_at']=_now()
        self.store.put('annotation',[record])
        candidate=self.candidate(record['stem'])
        candidate.pop('annotations',None)
        candidate['review_complete']=False
        candidate['status']='reviewed' if any(self.is_reviewed(r) for r in self._records() if r.get('stem')==record['stem']) else 'pending'
        self.store.put('candidate',[candidate])
        return {'withdrawn':True,'annotation_id':annotation_id,'history_preserved':True}

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
            image_path = self._candidate_path(record['stem'])
            if cv2.imread(str(image_path)) is None:
                issues.append({"type": "corrupt_or_missing_image", "path": str(image_path)})
            if not self.is_reviewed(record):
                issues.append({'type': 'unreviewed_annotation', 'annotation_id': record.get('annotation_id')})
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
        reviewed = [r for r in records if self.is_reviewed(r)]
        reviewed_detection = [r for r in reviewed if r.get('task','detect')=='detect']
        class_annotations = Counter(r["class_name"] for r in records)
        class_reviewed = Counter(r["class_name"] for r in reviewed_detection)
        pending_suggestions = Counter()
        pending_photos = Counter()
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
                        suggested_classes.add(name)
                pending_photos.update(suggested_classes)
        class_photos: dict[str, set[str]] = defaultdict(set)
        for r in records:
            class_photos[r["class_name"]].add(r["stem"])
        image_counts = {split: len({r['stem'] for r in reviewed if r['split'] == split}) for split in SPLITS}
        per_class = {name: {"photos": len(class_photos[name]), "annotations": class_annotations[name],
                            "reviewed": class_reviewed[name], "pending": pending_photos[name], "suggestions":pending_suggestions[name],
                            "target": minimum, "missing": max(0, minimum - class_reviewed[name]),
                            "ready": class_reviewed[name] >= minimum} for name in CLASSES}
        by_id={c['id']:c for c in candidates}
        for name in CLASSES:
            counts={split:len({r['stem'] for r in reviewed_detection if r['class_name']==name and r['split']==split and by_id.get(r['stem'],{}).get('review_complete')}) for split in SPLITS}
            per_class[name]['complete_photos_by_split']=counts
            per_class[name]['ready']=counts['train']>=minimum and counts['val']>0 and counts['test']>0
        try:
            root_display = str(self.root.relative_to(ROOT))
        except ValueError:
            root_display = str(self.root)
        return {"name": "VisionAR Own Dataset v1", "root": root_display, "classes": CLASSES,
                "images": image_counts, "total_images": sum(image_counts.values()), "annotations": len(records),
                "reviewed_annotations": len(reviewed), "pending_annotations": sum(not self.is_reviewed(r) and not r.get('withdrawn_at') for r in records),
                "unreviewed_suggestions":sum(pending_suggestions.values()),
                "candidate_images": len(candidates), "pending_candidates": sum(r.get("status") == "pending" for r in candidates),
                "by_class": {n: class_reviewed[n] for n in CLASSES}, "per_class": per_class,
                "reviewed_masks":len(reviewed)-len(reviewed_detection),
                "reviewed_detections":len(reviewed_detection),
                "ready_for_training": all(per_class[n]['ready'] for n in CLASSES),
                "minimum_recommended_per_class": minimum, "required_annotations": minimum * len(CLASSES)}

    @staticmethod
    def validate_polygon(points):
        polygon = np.asarray(points, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 500:
            raise ValueError('El contorno requiere entre 3 y 500 vértices')
        if not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
            raise ValueError('Contorno fuera de la imagen')
        if abs(cv2.contourArea(polygon.astype(np.float32))) < .00001:
            raise ValueError('Contorno sin área')
        # Reject segment intersections, including repeated vertices.
        def orientation(a, b, c):
            u,v=b-a,c-a
            return float(u[0]*v[1]-u[1]*v[0])
        for i in range(len(polygon)):
            a, b = polygon[i], polygon[(i+1) % len(polygon)]
            if np.linalg.norm(a-b) < 1e-6:
                raise ValueError('Vértices repetidos')
            for j in range(i+2, len(polygon)):
                if i == 0 and j == len(polygon)-1:
                    continue
                c, d = polygon[j], polygon[(j+1) % len(polygon)]
                if orientation(a,b,c)*orientation(a,b,d) <= 0 and orientation(c,d,a)*orientation(c,d,b) <= 0:
                    if np.maximum(np.minimum(a,b), np.minimum(c,d)).max() <= 1:  # bounding boxes below
                        if np.all(np.maximum(np.minimum(a,b), np.minimum(c,d)) <= np.minimum(np.maximum(a,b), np.maximum(c,d))):
                            raise ValueError('El contorno se cruza consigo mismo')
        return polygon.tolist()

    @transactional
    def approve_objects(self, candidate_id, objects, split, complete=False):
        if not isinstance(objects, list) or not 1 <= len(objects) <= 100:
            raise ValueError('Revisa entre 1 y 100 objetos por fotografía')
        saved = []
        for obj in objects:
            task = obj.get('task', 'detect')
            if task not in {'detect', 'segment'}:
                raise ValueError('Tarea no válida')
            polygon = self.validate_polygon(obj.get('polygon')) if task == 'segment' else None
            box = obj.get('box')
            if polygon:
                xs, ys = zip(*polygon)
                box = dict(x=min(xs), y=min(ys), w=max(xs)-min(xs), h=max(ys)-min(ys))
            result = self.approve_candidate(candidate_id, obj.get('class_name'), box, split)
            if polygon:
                records = self._records()
                record = next(r for r in records if r['annotation_id'] == result['annotation_id'])
                # Keep detection independently; masks have their own immutable record.
                record = dict(record, annotation_id=hashlib.sha256(json.dumps([candidate_id,obj.get('class_name'),polygon]).encode()).hexdigest()[:20],
                              task='segment', polygon=polygon, annotation='manual_polygon_reviewed')
                self.store.put('annotation', [record])
                result = record
            saved.append(result)
        candidate = self.candidate(candidate_id)
        candidate.pop('annotations',None)
        candidate['review_complete'] = bool(complete)
        self.store.put('candidate', [candidate])
        return {'saved': True, 'objects': saved, 'review_complete': bool(complete)}

    @transactional
    def reconcile(self):
        candidates, records = self._candidates(), self._records()
        issues = [{'type':'migration_error', **e} for e in self.store.errors()]
        by_id = {c['id']: c for c in candidates}
        hashes = defaultdict(list)
        for candidate in candidates:
            path = self._candidate_path(candidate['id'])
            if not path.exists():
                issues.append({'type':'missing_candidate', 'id':candidate['id']})
                continue
            raw = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                issues.append({'type':'corrupt_candidate', 'id':candidate['id']})
                continue
            digest = hashlib.sha256(raw).hexdigest()
            hashes[digest].append(candidate['id'])
            if digest != candidate.get('sha256'):
                issues.append({'type':'hash_mismatch', 'id':candidate['id']})
            candidate['physical_verified'] = digest == candidate.get('sha256')
            annotations = [r['annotation_id'] for r in records if r.get('stem') == candidate['id'] and self.is_reviewed(r)]
            candidate['approved_annotations'] = annotations
            if candidate.get('status') != 'rejected':
                candidate['status'] = 'reviewed' if annotations else 'pending'
            if not candidate.get('source') or not candidate.get('license'):
                issues.append({'type':'missing_provenance', 'id':candidate['id']})
            for name in candidate.get('intended_classes',[]):
                if name not in CLASSES:
                    issues.append({'type':'unknown_intended_class','id':candidate['id'],'class':name})
            for suggestion in candidate.get('suggestions',[]):
                try:
                    self._validate_box(suggestion.get('box',{}))
                except ValueError:
                    issues.append({'type':'invalid_suggested_box','id':candidate['id']})
            # A supplied URL is traceable evidence, not a verified redistribution permission.
            candidate['provenance_traceable'] = bool(re.search(r'https?://', candidate.get('source','')) and re.search(r'https?://',candidate.get('license','')))
        for path in self.candidate_dir.glob('*'):
            if path.is_file() and path.stem not in by_id:
                issues.append({'type':'orphan_candidate_file', 'file':path.name})
        for record in records:
            if record.get('stem') not in by_id:
                issues.append({'type':'orphan_annotation', 'annotation_id':record.get('annotation_id')})
        legacy_images=list((self.root/'images').glob('*/*'))
        legacy_labels=list((self.root/'labels').glob('*/*.txt'))
        referenced={(r.get('split'),r.get('stem')) for r in records}
        for path in legacy_images+legacy_labels:
            if path.is_file() and (path.parent.name,path.stem) not in referenced:
                issues.append({'type':'orphan_legacy_file','path':str(path.relative_to(self.root))})
        parents={c['id']:c['id'] for c in candidates}
        def find(key):
            while parents[key]!=key:
                parents[key]=parents[parents[key]]
                key=parents[key]
            return key
        for i,left in enumerate(candidates):
            for right in candidates[i+1:]:
                if left.get('sha256')==right.get('sha256') or (left.get('dhash') and right.get('dhash') and self._hamming(left['dhash'],right['dhash'])<=3):
                    parents[find(right['id'])]=find(left['id'])
        groups=defaultdict(list)
        for candidate in candidates:
            candidate['duplicate_group']=find(candidate['id'])
            groups[candidate['duplicate_group']].append(candidate['id'])
        self.store.put('candidate', candidates)
        return {'reconciled_at':_now(), 'candidates':len(candidates), 'physical_files':len(list(self.candidate_dir.glob('*.jpg'))),
                'states':dict(Counter(c.get('status','pending') for c in candidates)),
                'duplicates':[ids for ids in hashes.values() if len(ids)>1], 'near_duplicate_groups':[ids for ids in groups.values() if len(ids)>1], 'issues':issues,
                'traceable_provenance':sum(c.get('provenance_traceable',False) for c in candidates),
                'license_evidence_pending':sum(not c.get('provenance_traceable',False) for c in candidates),
                'legacy_images':len(legacy_images),'legacy_labels':len(legacy_labels),'stats':self.stats()}

    def training_audit(self, classes=None, task='detect', minimum=30):
        classes = list(dict.fromkeys(classes or CLASSES))
        if not classes or any(c not in CLASSES for c in classes) or task not in {'detect','segment'}:
            raise ValueError('Clases o tarea no válidas')
        candidates = {c['id']:c for c in self._candidates()}
        records = [r for r in self._records() if self.is_reviewed(r) and r.get('task','detect') == task and r['class_name'] in classes]
        issues = []
        counts = {c:{s:len({r['stem'] for r in records if r['class_name']==c and r['split']==s}) for s in sorted(SPLITS)} for c in classes}
        for c, splits in counts.items():
            if splits['train'] < minimum or splits['val'] < 1 or splits['test'] < 1:
                issues.append({'type':'insufficient_reviewed_images', 'class':c, 'counts':splits, 'minimum_train':minimum})
        checked = {}
        for r in records:
            c = candidates.get(r['stem'], {})
            if not c.get('review_complete'):
                issues.append({'type':'incomplete_image_review','id':r['stem']})
            if not c.get('source') or not c.get('license'):
                issues.append({'type':'missing_provenance','id':r['stem']})
            path = self._candidate_path(r['stem'])
            if r['stem'] not in checked:
                checked[r['stem']] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == r.get('sha256') and cv2.imread(str(path)) is not None
            if not checked[r['stem']]:
                issues.append({'type':'invalid_image','id':r['stem']})
            if task == 'segment':
                self.validate_polygon(r.get('polygon'))
            else:
                self._validate_box(r.get('box'))
        # Leakage is checked against ALL approved classes, not only the selection.
        all_records = [r for r in self._records() if self.is_reviewed(r)]
        for i, left in enumerate(all_records):
            for right in all_records[i+1:]:
                if left['split'] != right['split'] and (left.get('sha256') == right.get('sha256') or
                    (left.get('dhash') and right.get('dhash') and self._hamming(left['dhash'],right['dhash']) <= 3) or
                    (candidates.get(left['stem'],{}).get('group_id') and candidates[left['stem']]['group_id'] == candidates.get(right['stem'],{}).get('group_id')) or
                    (candidates.get(left['stem'],{}).get('duplicate_group') and candidates[left['stem']]['duplicate_group']==candidates.get(right['stem'],{}).get('duplicate_group'))):
                    issues.append({'type':'split_leakage','left':left['stem'],'right':right['stem']})
        return {'ready':bool(records) and not issues, 'issues':issues, 'counts':counts, 'classes':classes, 'task':task, 'annotations':len(records)}

    @transactional
    def export_snapshot(self, destination, classes=None, task='detect', minimum=30):
        # Recompute duplicate components in the same transaction as the snapshot.
        self.reconcile()
        audit = self.training_audit(classes, task, minimum)
        if not audit['ready']:
            raise ValueError('Dataset no apto: ' + json.dumps(audit['issues'], ensure_ascii=False)[:2000])
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        classes = audit['classes']
        records = [r for r in self._records() if self.is_reviewed(r) and r.get('task','detect')==task and r['class_name'] in classes]
        grouped = defaultdict(list)
        for r in records:
            grouped[(r['split'],r['stem'])].append(r)
        for (split, stem), objects in grouped.items():
            images, labels = destination/'images'/split, destination/'labels'/split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._candidate_path(stem), images/f'{stem}.jpg')
            rows=[]
            for obj in objects:
                b=obj['box']
                values = [v for p in obj['polygon'] for v in p] if task=='segment' else [b['x']+b['w']/2,b['y']+b['h']/2,b['w'],b['h']]
                rows.append(str(classes.index(obj['class_name']))+' '+' '.join(f'{v:.6f}' for v in values))
            (labels/f'{stem}.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8')
        config = {'path':destination.resolve().as_posix(), 'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(classes))}
        (destination/'data.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
        (destination/'annotations.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        return destination/'data.yaml'
