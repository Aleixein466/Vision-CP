"""API web local para fusionar cámara, YOLO26 y llama.cpp."""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from visual_fusion import VisualFusionEngine, crop_image_data_url
    from dataset_studio import DatasetStudio
    from training_manager import TrainingManager
except ModuleNotFoundError:
    from scripts.visual_fusion import VisualFusionEngine, crop_image_data_url
    from scripts.dataset_studio import DatasetStudio
    from scripts.training_manager import TrainingManager

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
MAX_BODY = 8 * 1024 * 1024
ERROR_LOG = ROOT / "training" / "metrics" / "vlm" / "web_errors.log"
SYSTEM_PROMPT = """Eres el asistente visual local de VisionAR AI. Responde únicamente en español.
Separa hechos visibles de interpretaciones y expresa incertidumbre cuando corresponda. No inventes marcas,
modelos, texto ilegible, identidades o detalles ocultos. No infieras género, etnia, salud, emoción interna ni
intenciones por la apariencia. Describe a las personas con lenguaje neutral: ropa, postura, accesorios y acciones
directamente observables. Las detecciones YOLO son evidencia orientativa, no verdad absoluta. Para conteos,
compáralas con la imagen. Si discrepan, dilo claramente."""


def llama_request(
    base_url: str,
    image_url: str,
    prompt: str,
    api_key: str | None = None,
    detection_context: str = "",
    history: list[dict] | None = None,
    model: str = "Qwen2-VL-2B-Instruct-GGUF",
) -> dict:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in (history or [])[-6:]:
        role = item.get("role")
        content = str(item.get("content", ""))[:1000]
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    grounded_prompt = prompt
    if detection_context:
        grounded_prompt += "\n\nEvidencia del detector:\n" + detection_context
    messages.append({"role": "user", "content": [
        {"type": "text", "text": grounded_prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]})
    payload = {
        "model": model, "messages": messages, "temperature": 0.15,
        "max_tokens": 180, "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as response:
        raw = json.loads(response.read().decode("utf-8"))
    answer = raw["choices"][0]["message"]["content"]
    return {"answer": answer.strip(), "usage": raw.get("usage", {}), "model": raw.get("model", model)}


def make_handler(llama_url: str, api_key: str | None = None, model_manager=None):
    deployed_model = ROOT / "training" / "checkpoints" / "visionar_38_best.pt"
    fusion = VisualFusionEngine(deployed_model if deployed_model.exists() else None)
    dataset = DatasetStudio()
    training = TrainingManager(dataset)
    threading.Thread(target=fusion.warmup, name="visionar-yolo-warmup", daemon=True).start()
    history: deque[dict] = deque(maxlen=30)
    history_lock = threading.Lock()
    vlm_lock = threading.Lock()
    detect_gate = threading.BoundedSemaphore(1)
    analysis_active = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB), **kwargs)

        def _json(self, status: int, value) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # El navegador puede cancelar un fotograma obsoleto; no es un fallo del modelo.
                return

        def _log_exception(self, exc: Exception) -> None:
            ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            with ERROR_LOG.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n[{datetime.now().astimezone().isoformat()}] {self.command} {self.path}: {exc}\n"
                    + traceback.format_exc()
                )

        def _request_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise ValueError("Solicitud ausente o mayor de 8 MB")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("El cuerpo JSON debe ser un objeto")
            return value

        def _health(self) -> dict:
            with urllib.request.urlopen(f"{llama_url}/health", timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            if route == "/api/status":
                try:
                    payload = {
                        "ready": True, "llama": self._health(), "yolo": "ready" if fusion.ready else "warming",
                        "yolo_model": {"scale": fusion.selected_scale, "task": fusion.selected_task},
                    }
                    if model_manager:
                        payload.update({"active_model": model_manager.selected, "metrics": model_manager.metrics()})
                    self._json(HTTPStatus.OK, payload)
                except Exception as exc:
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"ready": False, "error": str(exc)})
                return
            if route == "/api/models":
                models = model_manager.catalog() if model_manager else []
                self._json(HTTPStatus.OK, {"models": models, "llama_cpp": "b11879"})
                return
            if route == "/api/yolo/models":
                self._json(HTTPStatus.OK, {
                    "models": fusion.model_catalog(),
                    "selected": {"scale": fusion.selected_scale, "task": fusion.selected_task},
                })
                return
            if route == "/api/history":
                with history_lock:
                    items = list(history)
                self._json(HTTPStatus.OK, {"items": items})
                return
            if route == "/api/dataset/stats":
                self._json(HTTPStatus.OK, dataset.stats())
                return
            if route == "/api/dataset/candidates":
                self._json(HTTPStatus.OK, {"items": dataset.list_candidates(query.get("status", [None])[0])})
                return
            if route == "/api/dataset/candidate":
                self._json(HTTPStatus.OK, dataset.candidate(query.get("id", [""])[0], include_image=True))
                return
            if route == "/api/dataset/audit":
                self._json(HTTPStatus.OK, dataset.audit())
                return
            if route == "/api/training/status":
                self._json(HTTPStatus.OK, training.status())
                return
            if route == "/":
                self.path = "/index.html"
            return super().do_GET()

        def do_POST(self):
            try:
                if self.path == "/api/detect":
                    if analysis_active.is_set():
                        self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "YOLO en vivo está pausado mientras responde el VLM"})
                        return
                    request = self._request_json()
                    if not detect_gate.acquire(blocking=False):
                        self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Se descartó un fotograma porque la inferencia anterior continúa activa"})
                        return
                    try:
                        result = fusion.detect(
                            request.get("image", ""), request.get("confidence", 0.30), request.get("imgsz", 480),
                            "general", request.get("scale"), request.get("task"),
                        )
                    finally:
                        detect_gate.release()
                    self._json(HTTPStatus.OK, result)
                    return
                if self.path == "/api/yolo/select":
                    request = self._request_json()
                    with detect_gate:
                        selected = fusion.select_model(
                            str(request.get("scale", "n")), str(request.get("task", "detect")),
                            bool(request.get("download", False)),
                        )
                    self._json(HTTPStatus.OK, {"selected": selected, "models": fusion.model_catalog()})
                    return
                if self.path == "/api/dataset/sample":
                    request = self._request_json()
                    saved = dataset.add_sample(
                        request.get("image", ""), str(request.get("class_name", "")),
                        request.get("box"), str(request.get("split", "train")),
                        str(request.get("source", "web_camera")),
                    )
                    self._json(HTTPStatus.CREATED, {"sample": saved, "stats": dataset.stats()})
                    return
                if self.path == "/api/dataset/import":
                    request = self._request_json()
                    result = dataset.import_candidate(request.get("image", ""), str(request.get("filename", "image.jpg")),
                                                      str(request.get("source", "")), str(request.get("license", "")))
                    self._json(HTTPStatus.CREATED if result["imported"] else HTTPStatus.OK,
                               {"result": result, "stats": dataset.stats()})
                    return
                if self.path == "/api/dataset/import-url":
                    request = self._request_json()
                    url = str(request.get("url", "")).strip()
                    def public_dataset_host(host: str | None) -> bool:
                        return bool(host) and (host == "raw.githubusercontent.com" or
                                               host == "huggingface.co" or host.endswith(".huggingface.co") or
                                               host.endswith(".hf.co") or host.endswith(".xethub.hf.co") or
                                               host == "roboflow.com" or host.endswith(".roboflow.com"))
                    parsed_url = urlparse(url)
                    if parsed_url.scheme != "https" or not public_dataset_host(parsed_url.hostname):
                        raise ValueError("URL no autorizada; usa HTTPS de Hugging Face, Roboflow o GitHub Raw")
                    remote_request = urllib.request.Request(url, headers={"User-Agent": "VisionAR-Dataset-Studio/1.0"})
                    with urllib.request.urlopen(remote_request, timeout=30) as response:
                        final_host = urlparse(response.geturl()).hostname
                        if not public_dataset_host(final_host):
                            raise ValueError("La descarga redirigió a un dominio no autorizado")
                        content_type = response.headers.get_content_type()
                        if not content_type.startswith("image/"):
                            raise ValueError("La URL no apunta directamente a una imagen")
                        raw = response.read(MAX_BODY + 1)
                    if len(raw) > MAX_BODY:
                        raise ValueError("La imagen remota supera 8 MB")
                    image_url = f"data:{content_type};base64," + base64.b64encode(raw).decode("ascii")
                    result = dataset.import_candidate(image_url, Path(parsed_url.path).name or "public-image.jpg",
                                                      str(request.get("source") or url), str(request.get("license", "")))
                    self._json(HTTPStatus.CREATED if result["imported"] else HTTPStatus.OK,
                               {"result": result, "stats": dataset.stats()})
                    return
                if self.path == "/api/dataset/suggest":
                    request = self._request_json()
                    candidate = dataset.candidate(str(request.get("candidate_id", "")), include_image=True)
                    with detect_gate:
                        result = fusion.detect(candidate["image"], request.get("confidence", .15),
                                               request.get("imgsz", 640), "general",
                                               request.get("scale", "s"), request.get("task", "detect"))
                    updated = dataset.set_suggestions(candidate["id"], result.get("detections", []))
                    self._json(HTTPStatus.OK, {"candidate": updated, "detector": {
                        "model": result.get("model"), "latency_ms": result.get("latency_ms"),
                        "detections": len(result.get("detections", []))}})
                    return
                if self.path == "/api/dataset/approve":
                    request = self._request_json()
                    saved = dataset.approve_candidate(str(request.get("candidate_id", "")),
                                                       str(request.get("class_name", "")), request.get("box"),
                                                       str(request.get("split", "train")))
                    self._json(HTTPStatus.CREATED, {"annotation": saved, "stats": dataset.stats()})
                    return
                if self.path == "/api/dataset/reject":
                    request = self._request_json()
                    self._json(HTTPStatus.OK, {"candidate": dataset.reject_candidate(str(request.get("candidate_id", "")))})
                    return
                if self.path == "/api/training/start":
                    request = self._request_json()
                    result = training.start(request.get("epochs", 40), str(request.get("model", "training/checkpoints/yolo26n-objv1-seg.pt")))
                    self._json(HTTPStatus.ACCEPTED if result.get("started") else HTTPStatus.CONFLICT, result)
                    return
                if self.path == "/api/training/cancel":
                    self._request_json()
                    self._json(HTTPStatus.OK, training.cancel())
                    return
                if self.path == "/api/models/select":
                    if not model_manager:
                        raise RuntimeError("El administrador de modelos no está activo")
                    request = self._request_json()
                    model_manager.switch(str(request.get("model", "")))
                    self._json(HTTPStatus.OK, {"selected": model_manager.selected, "models": model_manager.catalog()})
                    return
                if self.path != "/api/analyze":
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Ruta no encontrada"})
                    return
                request = self._request_json()
                image_url = request.get("image", "")
                prompt = str(request.get("prompt", "")).strip()[:1000]
                if not image_url.startswith("data:image/"):
                    raise ValueError("Se requiere una imagen data URL")
                if not prompt:
                    prompt = "Describe los objetos visibles, su ubicación y las relaciones observables. No inventes."
                user_prompt = prompt
                if not vlm_lock.acquire(blocking=False):
                    self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "El VLM ya está procesando otra solicitud"})
                    return
                try:
                    analysis_active.set()
                    started = time.perf_counter()
                    mode = "general"
                    with detect_gate:
                        detection = fusion.detect(
                            image_url, request.get("confidence", 0.30), request.get("imgsz", 480), mode,
                            request.get("scale"), request.get("task"),
                        )
                    focus = request.get("focus")
                    vlm_image = crop_image_data_url(image_url, focus)
                    if focus:
                        prompt = (
                            "Analiza prioritariamente el recorte seleccionado por el usuario. Identifica el objeto pequeño "
                            "por sus rasgos visibles; si no es distinguible, dilo sin adivinar. " + prompt
                        )
                    elif mode == "small":
                        prompt = "Examina especialmente las manos y los objetos pequeños sostenidos. " + prompt
                    elif mode == "aquarium":
                        prompt = (
                            "La escena corresponde a un acuario. No llames aves a formas acuáticas; describe peces u otros "
                            "elementos solo cuando sean visualmente distinguibles. " + prompt
                        )
                    elif mode == "contents":
                        prompt = (
                            "La región puede contener una billetera, bolso o recipiente abierto. Enumera por separado solo "
                            "los contenidos realmente visibles, como billetes, monedas, tarjetas o llaves. No infieras saldo, "
                            "identidad, banco, números ni datos privados que no sean legibles y necesarios. " + prompt
                        )
                    active_key = model_manager.api_key if model_manager else api_key
                    model_name = model_manager.selected if model_manager else "Qwen2-VL-2B-Instruct-GGUF"
                    response = llama_request(
                        llama_url, vlm_image, prompt, active_key, fusion.prompt_context(detection),
                        request.get("history", []), model_name,
                    )
                    response.update({
                        "request_id": str(uuid.uuid4()), "frame_id": str(request.get("frame_id") or uuid.uuid4()),
                        "detections": detection["detections"], "counts": detection["counts"],
                        "yolo": {key: detection[key] for key in (
                            "latency_ms", "fps_equivalent", "model", "scale", "task",
                            "confidence_threshold", "imgsz"
                        )},
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "grounding": "YOLO y VLM combinados; las discrepancias requieren verificación",
                        "focus_used": bool(focus), "mode": mode,
                    })
                    record = {
                        "request_id": response["request_id"], "frame_id": response["frame_id"],
                        "timestamp": datetime.now().astimezone().isoformat(), "prompt": user_prompt,
                        "answer": response["answer"], "counts": response["counts"],
                        "elapsed_seconds": response["elapsed_seconds"],
                    }
                    with history_lock:
                        history.appendleft(record)
                    self._json(HTTPStatus.OK, response)
                finally:
                    analysis_active.clear()
                    vlm_lock.release()
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                self._json(HTTPStatus.BAD_GATEWAY, {"error": f"llama.cpp respondió {exc.code}", "detail": detail[:1000]})
            except Exception as exc:
                self._log_exception(exc)
                error_text = str(exc).lower()
                resource_error = (
                    isinstance(exc, (MemoryError, OSError))
                    or "not enough memory" in error_text
                    or "insufficient memory" in error_text
                    or "cannot create thread" in error_text
                )
                status = HTTPStatus.SERVICE_UNAVAILABLE if resource_error else HTTPStatus.BAD_GATEWAY
                message = "Recursos temporalmente insuficientes; se liberó el fotograma. Reduce la escala YOLO a Nano." if resource_error else f"No se pudo completar la operación: {exc}"
                self._json(status, {"error": message})

        def log_message(self, fmt, *args):
            print(f"[web] {self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--llama-url", default="http://127.0.0.1:8091")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.llama_url))
    print(f"VisionAR web: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
