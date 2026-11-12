"""Surya inference manager.

One process owns one SuryaInferenceManager. The manager wraps a single backend
(vllm | llamacpp) which speaks OpenAI-compatible chat completions.

Predictors take the manager via explicit injection at construction time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional

from surya.inference.backends.base import Backend
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _has_nvidia_gpu() -> bool:
    """True if an NVIDIA GPU is present on this host.

    Checks the kernel device node first (instant, load-independent, and avoids
    importing torch), then falls back to ``nvidia-smi -L``.
    """
    if os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia/version"):
        return True

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def _autodetect_backend() -> str:
    if settings.SURYA_INFERENCE_BACKEND:
        return settings.SURYA_INFERENCE_BACKEND
    # NVIDIA GPU → vllm, mps/cpu → llamacpp
    if _has_nvidia_gpu():
        return "vllm"
    return "llamacpp"


def _build_backend(method: str) -> Backend:
    method = method.lower()
    if method == "vllm":
        from surya.inference.backends.vllm import VllmBackend

        return VllmBackend()
    if method == "llamacpp":
        from surya.inference.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend()
    if method == "mlx":
        from surya.inference.backends.mlx import MlxBackend

        return MlxBackend()
    raise ValueError(
        f"Unknown inference backend {method!r}. Supported: 'vllm', 'llamacpp', 'mlx'."
    )


class SuryaInferenceManager:
    """Single entry point for VLM inference. Construct once per process."""

    def __init__(self, method: Optional[str] = None, lazy: bool = True):
        self.method = method or _autodetect_backend()
        self.backend: Backend = _build_backend(self.method)
        if not lazy:
            self.backend.start()

    def start(self) -> None:
        self.backend.start()

    def stop(self) -> None:
        self.backend.stop()

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        return self.backend.generate(batch)

    def capacity(self) -> int:
        """Server concurrency capacity (see Backend.capacity)."""
        return self.backend.capacity()


# Module-level lazy singleton for callers that don't want explicit construction
# (notebooks, ad-hoc scripts). Surya's own models.py and marker should use
# explicit construction.
_default_manager: Optional[SuryaInferenceManager] = None


def get_default_manager() -> SuryaInferenceManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SuryaInferenceManager()
    return _default_manager
