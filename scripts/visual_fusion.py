"""Detección, segmentación y metadatos verificables para VisionAR AI."""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import math
import shutil
import threading
import time
import io
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from runtime_limits import configure_runtime
except ModuleNotFoundError:
    from scripts.runtime_limits import configure_runtime

configure_runtime()

import cv2
import numpy as np
import psutil
from ultralytics import YOLO, YOLOE

try:
    from speech_announcer import SPANISH
except ModuleNotFoundError:
    from scripts.speech_announcer import SPANISH

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "training" / "checkpoints"
MODEL_PATH = CHECKPOINTS / "yolo26n-objv1-150.pt"
SEGMENT_MODEL_PATH = CHECKPOINTS / "yolo26n-objv1-seg.pt"
OPEN_MODEL_PATH = CHECKPOINTS / "yoloe-26n-seg-pf.pt"
CATALOG_PATH = ROOT / "CATALOGO_OBJETOS.json"
MANIFESTS = ROOT / "datasets" / "manifests"

SCALE_NAMES = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "Extra Large"}
TASK_NAMES = {"detect": "Detección", "segment": "Segmentación de instancias"}
CPU_GUIDANCE = {
    "n": "Recomendado para tiempo real en esta CPU",
    "s": "Más preciso; menor fluidez",
    "m": "Lento en la laptop objetivo",
    "l": "Muy lento en CPU",
    "x": "Solo para pruebas puntuales en CPU",
}
MIN_AVAILABLE_RAM_MB = {"n": 450, "s": 1200, "m": 2500, "l": 3200, "x": 5000}

SPECIAL_LABELS = {
    "person": "persona", "sneakers": "tenis", "chair": "silla", "hat": "gorra o sombrero",
    "bottle": "botella", "cabinet/shelf": "armario o estante", "cup": "taza",
    "glasses": "gafas", "desk": "escritorio", "handbag": "bolso", "book": "libro",
    "pillow": "almohada", "potted plant": "planta en maceta", "tv": "televisor",
    "fish": "pez", "goldfish": "pez dorado", "cat": "gato", "dog": "perro",
    "motorcycle": "moto", "fork": "tenedor", "spoon": "cuchara", "knife": "cuchillo",
    "backpack": "mochila", "pen/pencil": "lapicero o lápiz", "pencil case": "cartuchera",
    "marker": "marcador", "board eraser": "borrador de tablero", "eraser": "borrador",
    "tape": "cinta", "screwdriver": "destornillador", "notepaper": "hoja de papel",
    "dining table": "mesa", "coffee table": "mesa de centro", "couch": "sillón",
    "poker card": "carta de jugar", "balloon": "globo", "hotair balloon": "globo aerostático",
    "key": "llave", "keycard": "tarjeta llave", "connector": "conector",
    "cable": "cable", "electronic": "dispositivo electrónico", "mouse": "ratón de computador",
    "remote control": "control remoto", "cell phone": "teléfono celular", "smartphone": "teléfono celular",
    "watch": "reloj", "wallet": "billetera", "pouch": "estuche o billetera",
    "clutch": "bolso de mano", "clutch bag": "bolso de mano", "handbag": "bolso",
    "coin": "moneda", "scissors": "tijeras",
    "money": "dinero", "banknote": "billete", "cash": "dinero en efectivo",
    "credit card": "tarjeta de pago", "bank card": "tarjeta bancaria", "card": "tarjeta",
}
SMALL_OBJECT_NAMES = set(SPECIAL_LABELS)
AQUARIUM_TERMS = ("fish", "aquarium", "koi", "carp", "tetra", "guppy")
CONTENT_NAMES = {
    "wallet", "pouch", "clutch", "clutch bag", "handbag",
    "money", "banknote", "cash", "coin", "credit card", "bank card", "card",
    "key", "keycard", "usb drive", "usb flash drive", "flash drive",
}


def decode_image_data_url(data_url: str) -> np.ndarray:
    if not isinstance(data_url,str) or len(data_url)>12*1024*1024:
        raise ValueError('Imagen mayor de 8 MB o formato inválido')
    if not data_url.startswith("data:image/") or "," not in data_url:
        raise ValueError("Formato de imagen no válido")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except Exception as exc:
        raise ValueError("La imagen no contiene base64 válido") from exc
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as header:
        if header.width*header.height>16_000_000 or min(header.size)<16:
            raise ValueError('Dimensiones de imagen no permitidas')
        header.verify()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen")
    height, width = image.shape[:2]
    if width < 16 or height < 16 or width * height > 16_000_000:
        raise ValueError("Dimensiones de imagen no permitidas")
    return image


def crop_image_data_url(data_url: str, focus: dict | None, padding: float = 0.12) -> str:
    if not focus:
        return data_url
    image = decode_image_data_url(data_url)
    height, width = image.shape[:2]
    try:
        x, y, w, h = (float(focus[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Región seleccionada no válida") from exc
    if not all(0 <= value <= 1 for value in (x, y, w, h)) or w < 0.03 or h < 0.03 or x+w>1.00001 or y+h>1.00001:
        raise ValueError("La región seleccionada es demasiado pequeña o está fuera de la imagen")
    pad_x, pad_y = w * padding, h * padding
    x1, y1 = max(0, int((x - pad_x) * width)), max(0, int((y - pad_y) * height))
    x2, y2 = min(width, int((x + w + pad_x) * width)), min(height, int((y + h + pad_y) * height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("La región seleccionada está vacía")
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("No se pudo preparar la región seleccionada")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class VisualFusionEngine:
    """Mantiene un modelo YOLO seleccionado y serializa inferencias para proteger la CPU."""

    def __init__(self, model_path: Path | None = None):
        self.selected_scale = "n"
        self.selected_task = "detect"
        self._custom_path = model_path
        self._models: dict[tuple[str, str], YOLO] = {}
        self._custom_task='detect'
        self._custom_names={}
        if model_path is not None:
            custom=YOLO(str(model_path))
            self._custom_task=custom.task
            self._custom_names=custom.names
            self.selected_task=custom.task
            self._models[('n',custom.task)]=custom
        self._open_model: YOLOE | None = None
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self.ready = False
        self.knowledge = {}
        if CATALOG_PATH.exists():
            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            self.knowledge = {item["detector_class"]["name"]: item for item in catalog.get("records", [])}

    @staticmethod
    def filename(scale: str, task: str) -> str:
        if scale not in SCALE_NAMES or task not in TASK_NAMES:
            raise ValueError("Escala o tarea YOLO26 no reconocida")
        return f"yolo26{scale}{'-seg' if task == 'segment' else ''}.pt"

    def model_path(self, scale: str | None = None, task: str | None = None) -> Path:
        scale, task = scale or self.selected_scale, task or self.selected_task
        if self._custom_path is not None and scale == "n" and task == self._custom_task:
            return self._custom_path
        if scale == "n" and task == "detect":
            return MODEL_PATH
        if scale == "n" and task == "segment":
            return SEGMENT_MODEL_PATH
        if scale == "s" and task == "detect":
            return CHECKPOINTS / "yolo26s-objv1-150.pt"
        if scale == "s" and task == "segment":
            return CHECKPOINTS / "yolo26s-objv1-seg.pt"
        return CHECKPOINTS / self.filename(scale, task)

    def model_catalog(self) -> list[dict]:
        entries = []
        for task in TASK_NAMES:
            for scale in SCALE_NAMES:
                path = self.model_path(scale, task)
                entries.append({
                    "scale": scale, "scale_name": SCALE_NAMES[scale], "task": task,
                    "task_name": TASK_NAMES[task], "filename": path.name,
                    "vocabulary": (f"VisionAR personalizado ({len(self._custom_names)} clases)" if self._custom_path is not None and scale == "n" and task == self._custom_task
                                   else "Objects365 (365 clases)" if scale in {"n", "s"} else "COCO (80 clases)"),
                    "installed": path.exists(), "bytes": path.stat().st_size if path.exists() else None,
                    "active": scale == self.selected_scale and task == self.selected_task,
                    "cpu_guidance": CPU_GUIDANCE[scale],
                })
        return entries

    def _record_download(self, path: Path) -> None:
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {
            "retrieved_at": datetime.now().astimezone().isoformat(),
            "source": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{path.name}",
            "destination": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size, "sha256": digest,
        }
        (MANIFESTS / f"{path.name}.download.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _get_model(self, scale: str | None = None, task: str | None = None, allow_download: bool = False) -> YOLO:
        scale, task = scale or self.selected_scale, task or self.selected_task
        key = (scale, task)
        if key in self._models:
            return self._models[key]
        with self._load_lock:
            if key in self._models:
                return self._models[key]
            path = self.model_path(scale, task)
            if path.exists():
                model = YOLO(str(path))
            elif allow_download:
                filename = path.name
                model = YOLO(filename)
                downloaded = Path(model.ckpt_path).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                if downloaded != path.resolve():
                    shutil.move(str(downloaded), str(path))
                self._record_download(path)
            else:
                raise FileNotFoundError(f"El modelo {path.name} no está instalado")
            self._models[key] = model
            return model

    def _get_open_model(self) -> YOLOE:
        if self._open_model is None:
            with self._load_lock:
                if self._open_model is None:
                    if not OPEN_MODEL_PATH.exists():
                        raise FileNotFoundError(f"No se encontró {OPEN_MODEL_PATH}")
                    self._open_model = YOLOE(str(OPEN_MODEL_PATH))
        return self._open_model

    def select_model(self, scale: str, task: str, allow_download: bool = False) -> dict:
        with self._inference_lock:
            if scale not in SCALE_NAMES or task not in TASK_NAMES:
                raise ValueError("Escala o tarea YOLO26 no reconocida")
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            required_mb = MIN_AVAILABLE_RAM_MB[scale]
            if available_mb < required_mb:
                raise RuntimeError(
                    f"RAM insuficiente para YOLO26 {SCALE_NAMES[scale]}: "
                    f"disponible {available_mb:.0f} MB, mínimo de seguridad {required_mb} MB. Usa Nano."
                )
            if self._open_model is not None:
                self._open_model = None
                gc.collect()
            key = (scale, task)
            if key not in self._models:
                self._models.clear()
                gc.collect()
            model = self._get_model(scale, task, allow_download)
            model.predict(
                source=np.zeros((320, 320, 3), dtype=np.uint8), conf=0.32,
                imgsz=320, device="cpu", verbose=False,
            )
            self.selected_scale, self.selected_task = scale, task
            self._models = {key: model}
            self.ready = True
        return next(item for item in self.model_catalog() if item["active"])

    def warmup(self) -> None:
        self.select_model(self.selected_scale, self.selected_task, allow_download=False)

    def unload(self):
        with self._inference_lock:
            self._models.clear()
            self._open_model=None
            self.ready=False
            gc.collect()

    @staticmethod
    def presegment(image_url, box):
        if psutil.virtual_memory().available<600*2**20:
            raise ValueError('RAM insuficiente para presegmentar')
        image=decode_image_data_url(image_url)
        crop_image_data_url(image_url,box,padding=0)  # validate normalized rectangle
        scale=min(1,768/max(image.shape[:2]))
        image=cv2.resize(image,None,fx=scale,fy=scale)
        h,w=image.shape[:2]
        rect=(max(1,int(box['x']*w)),max(1,int(box['y']*h)),max(2,int(box['w']*w)-2),max(2,int(box['h']*h)-2))
        mask=np.zeros((h,w),np.uint8)
        cv2.grabCut(image,mask,rect,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_RECT)
        foreground=np.where((mask==1)|(mask==3),255,0).astype(np.uint8)
        contours,_=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError('No se encontró un contorno; dibújalo manualmente')
        contour=max(contours,key=cv2.contourArea)
        contour=cv2.approxPolyDP(contour,.003*cv2.arcLength(contour,True),True)
        polygon=[[round(float(p[0][0])/w,6),round(float(p[0][1])/h,6)] for p in contour]
        return {'polygon':polygon,'method':'grabcut','reviewed':False,'warning':'Propuesta de contorno; edita y revisa los límites antes de aprobar. No garantiza precisión anatómica.'}

    @staticmethod
    def map_region(result, focus):
        for item in result['detections']:
            b=item['box']
            item['box']={'x':focus['x']+b['x']*focus['w'],'y':focus['y']+b['y']*focus['h'], 'w':b['w']*focus['w'],'h':b['h']*focus['h']}
            if item.get('polygon'):
                item['polygon']=[[focus['x']+x*focus['w'],focus['y']+y*focus['h']] for x,y in item['polygon']]
        result['focus_used']=True
        return result

    @staticmethod
    def _polygon(result, result_index: int) -> list[list[float]] | None:
        if result.masks is None or result_index >= len(result.masks.xyn):
            return None
        points = result.masks.xyn[result_index].tolist()
        if not points:
            return None
        step = max(1, math.ceil(len(points) / 80))
        simplified = points[::step]
        return [[round(float(x), 5), round(float(y), 5)] for x, y in simplified]

    def detect(
        self, image_url: str, confidence: float = 0.30, imgsz: int = 480,
        mode: str = "general", scale: str | None = None, task: str | None = None,
        focus: dict | None = None,
    ) -> dict:
        if focus:
            result=self.detect(crop_image_data_url(image_url,focus,padding=0),confidence,max(imgsz,640),mode,scale,task)
            return self.map_region(result,focus)
        image = decode_image_data_url(image_url)
        height, width = image.shape[:2]
        if mode not in {"general", "small", "aquarium", "contents"}:
            raise ValueError("Modo de detección no reconocido")
        confidence = min(0.90, max(0.10, float(confidence)))
        imgsz = min(960, max(320, int(imgsz)))
        scale, task = scale or self.selected_scale, task or self.selected_task
        started = time.perf_counter()
        with self._inference_lock:
            if mode == "general":
                if self._open_model is not None:
                    self._open_model = None
                    gc.collect()
                model = self._get_model(scale, task)
            else:
                if self._models:
                    self._models.clear()
                    gc.collect()
                model = self._get_open_model()
            result = model.predict(source=image, conf=confidence, imgsz=imgsz, device="cpu", verbose=False)[0]
            if mode == "general":
                self.ready = True
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = []
        boxes = result.boxes
        if boxes is not None:
            rows = zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())
            for result_index, (xyxy, conf, class_id) in enumerate(rows):
                class_id = int(class_id)
                name = str(result.names[class_id])
                if mode == "small" and name.lower() not in SMALL_OBJECT_NAMES:
                    continue
                if mode == "aquarium" and not any(term in name.lower() for term in AQUARIUM_TERMS):
                    continue
                if mode == "contents" and name.lower() not in CONTENT_NAMES:
                    continue
                x1, y1, x2, y2 = xyxy
                singular = SPANISH.get(name, (name, name))[0]
                if name.lower() in SPECIAL_LABELS:
                    singular = SPECIAL_LABELS[name.lower()]
                elif mode == "aquarium":
                    singular = "pez" if "fish" in name.lower() else name
                detections.append({
                    "id": f"obj-{result_index + 1}", "class_id": class_id, "class": name,
                    "label": singular.removeprefix("un ").removeprefix("una ").removeprefix("unas "),
                    "confidence": round(float(conf), 4),
                    "uncertainty": 'tentative' if conf < .5 else 'detector_evidence',
                    "occlusion": 'unknown',
                    "description": self.knowledge.get(name, {}).get("description"),
                    "box": {
                        "x": round(x1 / width, 5), "y": round(y1 / height, 5),
                        "w": round((x2 - x1) / width, 5), "h": round((y2 - y1) / height, 5),
                    },
                    "polygon": self._polygon(result, result_index),
                })
        counts = Counter(item["label"] for item in detections)
        model_name = OPEN_MODEL_PATH.name if mode != "general" else self.model_path(scale, task).name
        return {
            "detections": detections, "counts": dict(counts),
            "image": {"width": width, "height": height},
            "latency_ms": round(elapsed_ms, 2),
            "fps_equivalent": round(1000 / elapsed_ms, 2) if elapsed_ms else None,
            "model": model_name, "scale": scale, "task": task,
            "confidence_threshold": confidence, "imgsz": imgsz, "mode": mode,
        }

    @staticmethod
    def prompt_context(result: dict) -> str:
        if not result["detections"]:
            return (
                "YOLO26 no detectó objetos por encima del umbral. Esto no demuestra que la escena esté vacía; "
                "analiza la imagen y expresa incertidumbre."
            )
        rows = [
            f"{item['id']}: {item['label']} (confianza {item['confidence']:.0%}, "
            f"caja normalizada x={item['box']['x']:.2f}, y={item['box']['y']:.2f}, "
            f"w={item['box']['w']:.2f}, h={item['box']['h']:.2f})"
            for item in result["detections"]
        ]
        return (
            "YOLO26 produjo estas detecciones orientativas, que pueden contener errores:\n- "
            + "\n- ".join(rows)
            + "\nÚsalas para conteos y ubicación, pero contrástalas con la imagen. Señala cualquier discrepancia. "
            "No repitas coordenadas ni porcentajes técnicos salvo que el usuario los solicite."
        )
