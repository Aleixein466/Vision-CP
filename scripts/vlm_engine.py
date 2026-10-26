"""Gestión segura de llama.cpp y modelos visuales instalados."""
from __future__ import annotations

import json
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

try:
    from runtime_limits import configure_runtime
except ModuleNotFoundError:
    from scripts.runtime_limits import configure_runtime

configure_runtime()

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "tools" / "llama-b11879"
LLAMA_EXE = LLAMA_DIR / "llama-server.exe"
MODEL_MANIFEST = ROOT / "datasets" / "manifests" / "vlm_models.json"
LOG_PATH = ROOT / "training" / "metrics" / "vlm" / "llama-server.log"

MODEL_SPECS = {
    "qwen2-vl-2b": {
        "name": "Qwen2-VL 2B Instruct", "hf": "ggml-org/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M",
        "repository": "ggml-org/Qwen2-VL-2B-Instruct-GGUF", "quality": "Mayor calidad en español",
    },
    "smolvlm-500m": {
        "name": "SmolVLM 500M Instruct", "hf": "ggml-org/SmolVLM-500M-Instruct-GGUF",
        "repository": "ggml-org/SmolVLM-500M-Instruct-GGUF", "quality": "Más rápido; español menos fiable",
    },
}


class VLMEngine:
    def __init__(self, port: int = 8091, selected: str = "qwen2-vl-2b"):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.selected = selected
        self.api_key: str | None = None
        self.process: subprocess.Popen | None = None
        self._log_handle = None
        self._lock = threading.RLock()
        self.external = False
        self.started_at: float | None = None

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return response.status == 200 and payload.get("status") in {"ok", "no slot available"}
        except Exception:
            return False

    def start(self, wait_seconds: int = 600) -> None:
        with self._lock:
            if self.ready():
                if self.process is None:
                    self.external = True
                    self.api_key = None
                return
            if not LLAMA_EXE.exists():
                raise FileNotFoundError(f"Falta {LLAMA_EXE}")
            required_mb=2200 if self.selected=='qwen2-vl-2b' else 900
            if psutil.virtual_memory().available < required_mb*2**20:
                raise RuntimeError(f'RAM insuficiente para iniciar el VLM; se requieren {required_mb} MB disponibles')
            self.external = False
            self.api_key = secrets.token_urlsafe(32)
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = LOG_PATH.open("a", encoding="utf-8")
            spec = MODEL_SPECS[self.selected]
            command = [
                str(LLAMA_EXE), "-hf", spec["hf"], "--host", "127.0.0.1", "--port", str(self.port),
                "--ctx-size", "4096", "--parallel", "1", "--threads", "1", "--threads-batch", "1",
                "--no-mmproj-offload", "--api-key", self.api_key,
            ]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                self.process = subprocess.Popen(
                    command, cwd=LLAMA_DIR, stdout=self._log_handle, stderr=subprocess.STDOUT, creationflags=flags
                )
            except Exception:
                self._log_handle.close()
                self._log_handle=None
                self.api_key=None
                raise
            self.started_at = time.time()
        for _ in range(wait_seconds):
            if self.ready():
                return
            if self.process and self.process.poll() is not None:
                code = self.process.returncode
                self.stop()
                raise RuntimeError(f"llama-server terminó con código {code}; revise {LOG_PATH}")
            time.sleep(1)
        self.stop()
        raise TimeoutError("El modelo no quedó listo dentro del tiempo permitido")

    def stop(self) -> None:
        with self._lock:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                self.process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            self.api_key = None
            self.started_at = None

    def switch(self, model_key: str) -> None:
        if model_key not in MODEL_SPECS:
            raise ValueError("Modelo no reconocido")
        with self._lock:
            if self.external:
                raise RuntimeError("No se puede cambiar un servidor llama.cpp iniciado fuera de VisionAR AI")
            if model_key == self.selected and self.ready():
                return
            self.stop()
            self.selected = model_key
            self.start()

    def catalog(self) -> list[dict]:
        recorded = {}
        if MODEL_MANIFEST.exists():
            manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
            recorded = {item["repository"]: item for item in manifest.get("models", [])}
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        items = []
        for key, spec in MODEL_SPECS.items():
            record = recorded.get(spec["repository"], {})
            files = record.get("files", [])
            cached_dir = cache / ("models--" + spec["repository"].replace("/", "--"))
            items.append({
                "id": key, "name": spec["name"], "repository": spec["repository"],
                "installed": cached_dir.exists(), "active": key == self.selected and self.ready(),
                "size_bytes": sum(int(f.get("bytes", 0)) for f in files),
                "vision": True, "note": spec["quality"],
            })
        return items

    def metrics(self) -> dict:
        process_metrics = None
        if self.process is not None and self.process.poll() is None:
            proc = psutil.Process(self.process.pid)
            memory = proc.memory_info()
            process_metrics = {
                "pid": self.process.pid,
                "ram_mb": round(memory.rss / 1024 / 1024, 1),
                "private_mb": round(getattr(memory, "private", 0) / 1024 / 1024, 1),
            }
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available_mb": round(psutil.virtual_memory().available / 1024 / 1024),
            "llama_process": process_metrics,
            "uptime_seconds": round(time.time() - self.started_at) if self.started_at else None,
        }
