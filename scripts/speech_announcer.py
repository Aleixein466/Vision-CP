"""Síntesis de voz offline y no bloqueante para el detector de cámara."""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import pyttsx3

SPANISH = {
    "person": ("una persona", "personas"),
    "bicycle": ("una bicicleta", "bicicletas"),
    "car": ("un automóvil", "automóviles"),
    "motorcycle": ("una motocicleta", "motocicletas"),
    "airplane": ("un avión", "aviones"),
    "bus": ("un autobús", "autobuses"),
    "train": ("un tren", "trenes"),
    "truck": ("un camión", "camiones"),
    "boat": ("un barco", "barcos"),
    "traffic light": ("un semáforo", "semáforos"),
    "fire hydrant": ("un hidrante de incendios", "hidrantes de incendios"),
    "stop sign": ("una señal de pare", "señales de pare"),
    "parking meter": ("un parquímetro", "parquímetros"),
    "bench": ("un banco", "bancos"),
    "bird": ("un pájaro", "pájaros"),
    "cat": ("un gato", "gatos"),
    "dog": ("un perro", "perros"),
    "horse": ("un caballo", "caballos"),
    "sheep": ("una oveja", "ovejas"),
    "cow": ("una vaca", "vacas"),
    "elephant": ("un elefante", "elefantes"),
    "bear": ("un oso", "osos"),
    "zebra": ("una cebra", "cebras"),
    "giraffe": ("una jirafa", "jirafas"),
    "backpack": ("una mochila", "mochilas"),
    "umbrella": ("un paraguas", "paraguas"),
    "handbag": ("un bolso", "bolsos"),
    "tie": ("una corbata", "corbatas"),
    "suitcase": ("una maleta", "maletas"),
    "frisbee": ("un disco volador", "discos voladores"),
    "skis": ("un par de esquís", "pares de esquís"),
    "snowboard": ("una tabla de snowboard", "tablas de snowboard"),
    "sports ball": ("una pelota deportiva", "pelotas deportivas"),
    "kite": ("una cometa", "cometas"),
    "baseball bat": ("un bate de béisbol", "bates de béisbol"),
    "baseball glove": ("un guante de béisbol", "guantes de béisbol"),
    "skateboard": ("una patineta", "patinetas"),
    "surfboard": ("una tabla de surf", "tablas de surf"),
    "tennis racket": ("una raqueta de tenis", "raquetas de tenis"),
    "bottle": ("una botella", "botellas"),
    "wine glass": ("una copa", "copas"),
    "cup": ("una taza", "tazas"),
    "fork": ("un tenedor", "tenedores"),
    "knife": ("un cuchillo", "cuchillos"),
    "spoon": ("una cuchara", "cucharas"),
    "bowl": ("un tazón", "tazones"),
    "banana": ("un banano", "bananos"),
    "apple": ("una manzana", "manzanas"),
    "sandwich": ("un sándwich", "sándwiches"),
    "orange": ("una naranja", "naranjas"),
    "broccoli": ("un brócoli", "brócolis"),
    "carrot": ("una zanahoria", "zanahorias"),
    "hot dog": ("un perro caliente", "perros calientes"),
    "pizza": ("una pizza", "pizzas"),
    "donut": ("una dona", "donas"),
    "cake": ("un pastel", "pasteles"),
    "chair": ("una silla", "sillas"),
    "couch": ("un sofá", "sofás"),
    "potted plant": ("una planta en maceta", "plantas en maceta"),
    "bed": ("una cama", "camas"),
    "dining table": ("una mesa de comedor", "mesas de comedor"),
    "toilet": ("un inodoro", "inodoros"),
    "tv": ("un televisor", "televisores"),
    "laptop": ("un computador portátil", "computadores portátiles"),
    "mouse": ("un ratón de computador", "ratones de computador"),
    "remote": ("un control remoto", "controles remotos"),
    "keyboard": ("un teclado", "teclados"),
    "cell phone": ("un teléfono celular", "teléfonos celulares"),
    "microwave": ("un microondas", "microondas"),
    "oven": ("un horno", "hornos"),
    "toaster": ("una tostadora", "tostadoras"),
    "sink": ("un fregadero", "fregaderos"),
    "refrigerator": ("un refrigerador", "refrigeradores"),
    "book": ("un libro", "libros"),
    "clock": ("un reloj", "relojes"),
    "vase": ("un florero", "floreros"),
    "scissors": ("unas tijeras", "tijeras"),
    "teddy bear": ("un oso de peluche", "osos de peluche"),
    "hair drier": ("un secador de cabello", "secadores de cabello"),
    "toothbrush": ("un cepillo de dientes", "cepillos de dientes"),
}


def spanish_phrase(label: str, count: int = 1) -> str:
    singular, plural = SPANISH.get(label, (f"un objeto {label}", f"objetos {label}"))
    return f"Veo {singular}" if count == 1 else f"Veo {count} {plural}"


class SpeechAnnouncer:
    def __init__(self, rate: int, volume: float, log_path: Path):
        self.rate = rate; self.volume = volume; self.log_path = log_path
        self.messages: queue.Queue[str | None] = queue.Queue(maxsize=10)
        self.ready = threading.Event(); self.error: Exception | None = None
        self.thread = threading.Thread(target=self._worker, name="visionar-tts", daemon=True)
        self.thread.start(); self.ready.wait(timeout=10)
        if self.error: raise RuntimeError(f"No se pudo iniciar la voz: {self.error}")

    def _worker(self) -> None:
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate); engine.setProperty("volume", self.volume)
            for voice in engine.getProperty("voices"):
                text = " ".join([str(voice.id), str(voice.name), *(str(x) for x in getattr(voice, "languages", []))]).lower()
                if any(token in text for token in ("spanish", "español", "es-es", "es_")):
                    engine.setProperty("voice", voice.id); break
            self.ready.set()
            while True:
                message = self.messages.get()
                try:
                    if message is None: break
                    engine.say(message); engine.runAndWait()
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"timestamp": datetime.now().astimezone().isoformat(), "message": message}, ensure_ascii=False) + "\n")
                finally:
                    self.messages.task_done()
        except Exception as exc:
            self.error = exc; self.ready.set()
        finally:
            if engine is not None: engine.stop()

    def say_object(self, label: str, count: int = 1) -> bool:
        try:
            self.messages.put_nowait(spanish_phrase(label, count)); return True
        except queue.Full:
            return False

    def say(self, message: str) -> bool:
        try:
            self.messages.put_nowait(message); return True
        except queue.Full:
            return False

    def flush(self) -> None:
        self.messages.join()

    def close(self) -> None:
        if self.thread.is_alive():
            self.messages.put(None); self.thread.join(timeout=10)

