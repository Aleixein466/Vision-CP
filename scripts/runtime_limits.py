"""Limites conservadores para inferencia local en Windows/CPU."""
from __future__ import annotations

import os


def configure_runtime() -> None:
    # Deben existir antes de que NumPy, OpenCV o PyTorch creen sus pools.
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        import cv2
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except (ImportError, AttributeError):
        pass

    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch solo permite fijarlo antes del primer trabajo paralelo.
            pass
    except ImportError:
        pass
