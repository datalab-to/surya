"""MLX backend (Apple Silicon): spawns `mlx_vlm.server`, an OpenAI-compatible server, and talks to it like the other backends.

Install:    pip install mlx-vlm==0.7.10      (Apple Silicon only; the tested version)
Weights:    an MLX checkpoint dir or HF repo, set as SURYA_MLX_MODEL. Build one from a Surya checkpoint with
            `python surya/scripts/build_mlx.py` (it also repairs the tokenizer files and the end-token id).
Select it:  SURYA_INFERENCE_BACKEND=mlx   (auto-detection still picks vllm / llamacpp)
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.backends.openai_client import chat_completions_batch
from surya.inference.backends.spawn import (
    SpawnError,
    SpawnHandle,
    attach_or_spawn,
    probe_health,
    probe_model_id,
)
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _health_url(port: int) -> str:
    return f"http://{settings.SURYA_INFERENCE_HOST}:{port}"


def _openai_url(port: int) -> str:
    return f"http://{settings.SURYA_INFERENCE_HOST}:{port}/v1"


def _require_mlx_vlm() -> None:
    if importlib.util.find_spec("mlx_vlm") is None:
        raise SpawnError(
            "mlx-vlm is not installed. Install it (Apple Silicon only) with:\n"
            "  pip install mlx-vlm==0.7.10\n"
            "or point SURYA_INFERENCE_URL at a running mlx_vlm.server."
        )


class MlxBackend(Backend):
    name = "mlx"
    # mlx_vlm.server batches concurrent requests; this keeps a few in flight without
    # growing KV memory much on a laptop.
    DEFAULT_PARALLEL = 4

    def __init__(self):
        self.handle: Optional[ServerHandle] = None
        self._client: Optional[OpenAI] = None

    def _attach_external(self) -> ServerHandle:
        """SURYA_INFERENCE_URL pinned: attach without spawning. The server reports its own model id (the --model value)."""
        base = settings.SURYA_INFERENCE_URL.rstrip("/")
        health = base[: -len("/v1")] if base.endswith("/v1") else base
        if not probe_health(health):
            raise SpawnError(
                f"SURYA_INFERENCE_URL={base} is not reachable at /health. "
                "Start mlx_vlm.server or unset the variable."
            )
        reported = probe_model_id(base)
        if not reported:
            raise SpawnError(
                f"{base} did not report a model id at /v1/models; is it an mlx_vlm.server?"
            )
        wanted = settings.SURYA_MLX_MODEL
        if wanted and wanted not in (reported, Path(reported).name, Path(wanted).name):
            raise SpawnError(
                f"{base} serves {reported!r} but SURYA_MLX_MODEL={wanted!r}. Unset one or restart the server."
            )
        model = reported
        return ServerHandle(base_url=base, model_name=model, spawned_by_us=False)

    def start(self) -> ServerHandle:
        if self.handle is not None:
            return self.handle

        if settings.SURYA_INFERENCE_URL:
            self.handle = self._attach_external()
            self._client = OpenAI(api_key="EMPTY", base_url=self.handle.base_url)
            return self.handle

        model = settings.SURYA_MLX_MODEL
        if not model:
            raise SpawnError(
                "SURYA_MLX_MODEL is not set. Point it at an MLX checkpoint (a local dir or an HF repo). "
                "Build one with `python surya/scripts/build_mlx.py --checkpoint <repo-or-dir> --q-bits 8`."
            )
        _require_mlx_vlm()
        if settings.SURYA_STOP_STRINGS:
            logger.warning(
                "mlx_vlm.server ignores `stop`; SURYA_STOP_STRINGS has no effect on the MLX backend. "
                "The model must end its own output (real <|im_end|>)."
            )

        def spawn_fn(port: int) -> SpawnHandle:
            cmd = [
                sys.executable,
                "-m",
                "mlx_vlm.server",
                "--model",
                model,
                "--host",
                settings.SURYA_INFERENCE_HOST,
                "--port",
                str(port),
            ]
            cmd += shlex.split(settings.MLX_VLM_EXTRA_ARGS or "")
            logger.info(f"Spawning: {' '.join(cmd)}")
            log_path = Path("~/.cache/datalab/surya/mlx_server.log").expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(
                log_path, "ab"
            ) as log_fp:  # the child keeps its own copy of the descriptor
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            return SpawnHandle(
                pid=proc.pid, cleanup_id=str(proc.pid), cleanup_kind="process"
            )

        spawned = attach_or_spawn(
            backend=self.name,
            expected_model_name=model,  # mlx_vlm.server reports the --model value as the model id
            spawn_fn=spawn_fn,
            health_url_for=_health_url,
            openai_url_for=_openai_url,
            startup_timeout=settings.SURYA_INFERENCE_STARTUP_TIMEOUT,
        )
        self.handle = ServerHandle(
            base_url=spawned.base_url,
            model_name=spawned.model_name,
            spawned_by_us=spawned.spawned_by_us,
        )
        self._client = OpenAI(api_key="EMPTY", base_url=self.handle.base_url)
        return self.handle

    def capacity(self) -> int:
        return settings.SURYA_INFERENCE_PARALLEL or self.DEFAULT_PARALLEL

    def stop(self) -> None:
        # atexit handler in spawn.py owns cleanup; nothing to do here.
        self.handle = None
        self._client = None

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        if self.handle is None or self._client is None:
            self.start()
        return chat_completions_batch(
            batch,
            client=self._client,
            model_name=self.handle.model_name,
            timeout=settings.SURYA_INFERENCE_TIMEOUT_SECONDS,
            max_workers=self.capacity(),
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
            allow_stop_token_ids=False,  # vllm-only request extension
        )
