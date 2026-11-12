from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


from surya.inference.schema import BatchInputItem, BatchOutputItem


@dataclass
class ServerHandle:
    base_url: str  # e.g. "http://127.0.0.1:8765/v1"
    model_name: str  # what gets passed in OpenAI `model` field
    spawned_by_us: bool  # if True, we manage atexit cleanup
    cleanup_fn: Optional[Callable[[], None]] = None


class Backend:
    """Abstract backend. Concrete backends own server lifecycle + generation."""

    name: str  # "vllm" | "llamacpp"

    def start(self) -> ServerHandle:
        """Idempotent: probe → attach if alive, else spawn. Returns handle."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the server if we spawned it."""
        raise NotImplementedError

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        raise NotImplementedError

    def capacity(self) -> int:
        """Server-side concurrency capacity (concurrent requests the server can
        actively process). Callers sizing multi-process client pools should
        target an aggregate in-flight count of ~this value."""
        return 8
