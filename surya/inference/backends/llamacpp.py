"""llama.cpp backend: spawns the upstream `llama-server` binary natively.

Install:
- macOS:    brew install llama.cpp     (Metal build, MPS)
- Linux:    brew install llama.cpp  OR  github.com/ggml-org/llama.cpp/releases
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from huggingface_hub import hf_hub_download
from openai import OpenAI

from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.backends.openai_client import chat_completions_batch
from surya.inference.backends.spawn import (
    SpawnHandle,
    SpawnError,
    attach_or_spawn,
)
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _resolve_llama_server_binary() -> str:
    binary = settings.LLAMA_CPP_BINARY
    if binary and os.path.isfile(binary):
        return binary
    found = shutil.which(binary or "llama-server")
    if found:
        return found
    raise SpawnError(
        "llama-server binary not found. Install with:\n"
        "  macOS:  brew install llama.cpp\n"
        "  Linux:  brew install llama.cpp  OR download from\n"
        "          https://github.com/ggml-org/llama.cpp/releases\n"
        "Or set LLAMA_CPP_BINARY in your env to the binary path."
    )


def _download_gguf_files() -> tuple[str, str]:
    """Download model + mmproj GGUFs from HF Hub. Returns local paths."""
    repo = settings.SURYA_GGUF_REPO
    model_file = settings.SURYA_GGUF_MODEL_FILE
    mmproj_file = settings.SURYA_GGUF_MMPROJ_FILE
    logger.info(f"Downloading {model_file} and {mmproj_file} from {repo}")
    model_path = hf_hub_download(repo_id=repo, filename=model_file)
    mmproj_path = hf_hub_download(repo_id=repo, filename=mmproj_file)
    return model_path, mmproj_path


def _health_url(port: int) -> str:
    return f"http://{settings.SURYA_INFERENCE_HOST}:{port}"


def _openai_url(port: int) -> str:
    return f"http://{settings.SURYA_INFERENCE_HOST}:{port}/v1"


class LlamaCppBackend(Backend):
    name = "llamacpp"
    # Conservative default slot count - each parallel slot consumes KV cache,
    # so llama.cpp can't fan out as wide as a server-class GPU under vllm.
    DEFAULT_PARALLEL = 8

    def __init__(self):
        self.handle: Optional[ServerHandle] = None
        self._client: Optional[OpenAI] = None

    def start(self) -> ServerHandle:
        if self.handle is not None:
            return self.handle

        # If user pinned an external server, attach without spawning.
        # No binary or GGUF download needed in that case.
        if settings.SURYA_INFERENCE_URL:
            spawned = attach_or_spawn(
                backend=self.name,
                expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
                spawn_fn=lambda port: SpawnHandle(
                    pid=None, cleanup_id="", cleanup_kind="process"
                ),  # never called
                health_url_for=_health_url,
                openai_url_for=_openai_url,
                startup_timeout=settings.SURYA_INFERENCE_STARTUP_TIMEOUT,
            )
            self.handle = ServerHandle(
                base_url=spawned.base_url,
                model_name=spawned.model_name,
                spawned_by_us=spawned.spawned_by_us,
            )
            self._client = OpenAI(
                api_key=settings.SURYA_INFERENCE_API_KEY,
                base_url=self.handle.base_url,
            )
            return self.handle

        binary = _resolve_llama_server_binary()

        # Pre-download GGUFs so the spawn doesn't race the download
        if (
            settings.SURYA_GGUF_LOCAL_MODEL_PATH
            and settings.SURYA_GGUF_LOCAL_MMPROJ_PATH
        ):
            model_path = settings.SURYA_GGUF_LOCAL_MODEL_PATH
            mmproj_path = settings.SURYA_GGUF_LOCAL_MMPROJ_PATH
        else:
            model_path, mmproj_path = _download_gguf_files()

        # Total KV-cache budget. llama-server divides --ctx-size across
        # --parallel slots, so a too-small total silently truncates outputs
        # once each slot's share fills. Scale with parallel by default;
        # SURYA_INFERENCE_CTX_SIZE overrides to a fixed value if set.
        parallel = settings.SURYA_INFERENCE_PARALLEL or self.DEFAULT_PARALLEL
        per_slot = settings.SURYA_INFERENCE_CTX_PER_SLOT
        ctx_size = settings.SURYA_INFERENCE_CTX_SIZE
        if ctx_size is None:
            ctx_size = max(16384, parallel * per_slot)
        effective_per_slot = ctx_size // max(parallel, 1)
        logger.info(
            f"llama-server ctx-size={ctx_size} "
            f"(~{effective_per_slot}/slot × {parallel} parallel slots)"
        )
        if effective_per_slot < per_slot:
            logger.warning(
                f"per-slot ctx ({effective_per_slot}) is below recommended "
                f"{per_slot}; outputs may truncate. Raise "
                f"SURYA_INFERENCE_CTX_SIZE or SURYA_INFERENCE_CTX_PER_SLOT, "
                f"or lower SURYA_INFERENCE_PARALLEL."
            )

        def spawn_fn(port: int) -> SpawnHandle:
            cmd = [
                binary,
                "-m",
                model_path,
                "--mmproj",
                mmproj_path,
                "-ngl",
                str(settings.LLAMA_CPP_NGL),
                "--host",
                settings.SURYA_INFERENCE_HOST,
                "--port",
                str(port),
                "--parallel",
                str(parallel),
                "--ctx-size",
                str(ctx_size),
                "--no-mmproj-offload" if settings.LLAMA_CPP_NO_MMPROJ_OFFLOAD else "",
                "--alias",
                settings.SURYA_MODEL_CHECKPOINT,
                "--jinja",
            ]
            cmd = [c for c in cmd if c]
            for extra in (settings.LLAMA_CPP_EXTRA_ARGS or "").split():
                cmd.append(extra)
            logger.info(f"Spawning: {' '.join(cmd)}")
            log_path = Path("~/.cache/datalab/surya/llamacpp_server.log").expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_path, "ab")
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
            expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
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
        self._client = OpenAI(
            api_key=settings.SURYA_INFERENCE_API_KEY,
            base_url=self.handle.base_url,
        )
        return self.handle

    def capacity(self) -> int:
        from surya.settings import settings

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
            max_workers=settings.SURYA_INFERENCE_PARALLEL or self.DEFAULT_PARALLEL,
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
        )
