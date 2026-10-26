"""Safe subprocess manager for the user-authorized VisionAR training job."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import torch

try:
    from dataset_studio import DatasetStudio
except ModuleNotFoundError:
    from scripts.dataset_studio import DatasetStudio

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "training" / "metrics" / "visionar_training_state.json"
LOG = ROOT / "training" / "metrics" / "visionar_training.log"


class TrainingManager:
    def __init__(self, dataset: DatasetStudio):
        self.dataset = dataset
        self.process: subprocess.Popen | None = None
        self.lock = threading.RLock()

    def _persist(self, data: dict) -> None:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def status(self) -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            saved = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
            saved.update({"running": running, "pid": self.process.pid if running else None,
                          "hardware": {"device": "cuda:0" if torch.cuda.is_available() else "cpu",
                                       "cuda_available": torch.cuda.is_available()}})
            if self.process is not None and not running:
                saved["returncode"] = self.process.returncode
            if LOG.exists():
                saved["log_tail"] = LOG.read_text(encoding="utf-8", errors="replace")[-6000:]
            return saved

    def start(self, epochs: int = 40, model: str = "training/checkpoints/yolo26n-objv1-seg.pt") -> dict:
        audit = self.dataset.audit(30)
        if not audit["ready"]:
            return {"started": False, "reason": "dataset_not_ready", "audit": audit}
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return {"started": False, "reason": "already_running", "status": self.status()}
            LOG.parent.mkdir(parents=True, exist_ok=True)
            log_handle = LOG.open("w", encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts" / "train_visionar.py"), "--execute",
                       "--epochs", str(max(1, min(int(epochs), 300))), "--model", model]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
                                            text=True, creationflags=flags)
            state = {"started": True, "started_at": datetime.now().astimezone().isoformat(),
                     "pid": self.process.pid, "command": command, "audit": audit}
            self._persist(state)
            return state

    def cancel(self) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return {"cancelled": False, "reason": "not_running"}
            pid = self.process.pid
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            state = self.status()
            state.update({"cancelled": True, "cancelled_at": datetime.now().astimezone().isoformat(), "pid": pid})
            self._persist(state)
            return state
