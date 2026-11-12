"""llama.cpp backend: spawns the upstream `llama-server` binary natively.

Install:
- macOS:    brew install llama.cpp     (Metal build, MPS)
- Linux:    brew install llama.cpp  OR  github.com/ggml-org/llama.cpp/releases
"""

from __future__ import annotations

import errno
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from typing import List, Optional

from huggingface_hub import hf_hub_download
from openai import OpenAI

from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.backends.openai_client import (
    _END_MARKERS,
    _strip_end_marker,
    chat_completions_batch,
)
from surya.inference.backends.spawn import (
    SpawnError,
    SpawnHandle,
    _cache_dir,
    attach_or_spawn,
)
from surya.inference.prompts import PROMPT_MAPPING
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.inference.util import scale_to_fit
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

_DEFAULT_POPEN = subprocess.Popen


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
    """Resolve model + mmproj GGUFs from local HF cache first, then HF Hub."""
    repo = settings.SURYA_GGUF_REPO
    model_file = settings.SURYA_GGUF_MODEL_FILE
    mmproj_file = settings.SURYA_GGUF_MMPROJ_FILE
    try:
        model_path = hf_hub_download(
            repo_id=repo, filename=model_file, local_files_only=True
        )
        mmproj_path = hf_hub_download(
            repo_id=repo, filename=mmproj_file, local_files_only=True
        )
        return model_path, mmproj_path
    except Exception:
        logger.info(f"Downloading {model_file} and {mmproj_file} from {repo}")
        model_path = hf_hub_download(repo_id=repo, filename=model_file)
        mmproj_path = hf_hub_download(repo_id=repo, filename=mmproj_file)
        return model_path, mmproj_path


def _loopback_blocked() -> bool:
    """Return True if the OS sandbox forbids outbound loopback TCP connections (EPERM)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", 9)) == errno.EPERM
    except OSError as exc:
        return exc.errno == errno.EPERM


def _apply_backspaces(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\x08":
            if out:
                out.pop()
        else:
            out.append(ch)
    return "".join(out)


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
        self._cli_binary: Optional[str] = None
        self._model_path: Optional[str] = None
        self._mmproj_path: Optional[str] = None

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

        # If running inside an OS sandbox that blocks loopback TCP connections
        # (EPERM), fall back to running `llama-cli` over standard pipes.
        if subprocess.Popen is _DEFAULT_POPEN and _loopback_blocked():
            cli_bin = shutil.which("llama-cli")
            if cli_bin:
                logger.info(
                    "Loopback TCP sockets are blocked by OS sandbox; using llama-cli stdio backend."
                )
                self._cli_binary = cli_bin
                self._model_path = model_path
                self._mmproj_path = mmproj_path
                self.handle = ServerHandle(
                    base_url="stdio://llama-cli",
                    model_name=settings.SURYA_MODEL_CHECKPOINT,
                    spawned_by_us=False,
                )
                return self.handle

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
                "--special" if settings.LLAMA_CPP_SPECIAL else "",
            ]
            cmd = [c for c in cmd if c]
            for extra in shlex.split(settings.LLAMA_CPP_EXTRA_ARGS or ""):
                cmd.append(extra)
            logger.info(f"Spawning: {' '.join(cmd)}")
            log_path = _cache_dir() / "llamacpp_server.log"
            with open(log_path, "ab") as log_fp:
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
            cleanup_fn=getattr(spawned, "cleanup_fn", None),
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
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        if self.handle is not None and self.handle.cleanup_fn is not None:
            try:
                self.handle.cleanup_fn()
            except Exception:
                pass
        self.handle = None
        self._client = None
        self._cli_binary = None

    def _generate_via_llama_cli(
        self, batch: List[BatchInputItem]
    ) -> List[BatchOutputItem]:
        if not batch:
            return []
        with tempfile.TemporaryDirectory(
            dir=_cache_dir(), prefix="cli_batch_"
        ) as tmpdir:
            lines: list[str] = []
            for idx, item in enumerate(batch):
                img_path = os.path.join(tmpdir, f"img_{idx}.png")
                scale_to_fit(item.image).save(
                    img_path, format="PNG", compress_level=1
                )
                prompt = (
                    item.prompt or PROMPT_MAPPING[item.prompt_type]
                ).replace("\n", " ").strip()
                if idx > 0:
                    lines.append("/clear")
                lines.append(f"/image {img_path}")
                lines.append(prompt)
            lines.append("/exit")
            script = "\n".join(lines) + "\n"

            max_tokens = max((item.max_tokens or 2048) for item in batch)
            temp = next(
                (item.temperature for item in batch if item.temperature is not None),
                0.0,
            )
            cmd = [
                self._cli_binary,
                "-m",
                self._model_path,
                "--mmproj",
                self._mmproj_path,
                "-ngl",
                str(settings.LLAMA_CPP_NGL),
                "--ctx-size",
                str(max(16384, max_tokens + 4096)),
                "--no-mmproj-offload" if settings.LLAMA_CPP_NO_MMPROJ_OFFLOAD else "",
                "--jinja",
                "--special" if settings.LLAMA_CPP_SPECIAL else "",
                "--temp",
                str(temp),
                "-n",
                str(max_tokens),
                "--no-warmup",
            ]
            cmd = [c for c in cmd if c]
            try:
                r = subprocess.run(
                    cmd,
                    input=script,
                    capture_output=True,
                    text=True,
                    timeout=settings.SURYA_INFERENCE_TIMEOUT_SECONDS,
                    start_new_session=True,
                )
            except Exception as e:
                logger.warning(f"llama-cli inference error: {e}")
                return [
                    BatchOutputItem(
                        raw="",
                        token_count=0,
                        error=True,
                        metadata=item.metadata,
                    )
                    for item in batch
                ]

        cleaned = _apply_backspaces(r.stdout or "")
        segments = cleaned.split("Loaded media from ")
        turn_segments = segments[1:] if len(segments) > 1 else []

        results: List[BatchOutputItem] = []
        for idx, item in enumerate(batch):
            if idx >= len(turn_segments):
                results.append(
                    BatchOutputItem(
                        raw="",
                        token_count=0,
                        error=True,
                        metadata=item.metadata,
                    )
                )
                continue
            seg = turn_segments[idx]
            m = re.search(r"\n+>\s+[^\n]+\n+(.*)", seg, re.DOTALL)
            body = m.group(1) if m else seg
            if "\n\n[ Prompt:" in body:
                body = body.split("\n\n[ Prompt:", 1)[0]
            body_r = body.rstrip()
            stopped = any(body_r.endswith(marker) for marker in _END_MARKERS)
            raw = _strip_end_marker(body_r).strip()
            finish_reason = "stop" if stopped else "length"
            token_count = max(1, len(raw) // 4) if raw else 0
            results.append(
                BatchOutputItem(
                    raw=raw,
                    token_count=token_count,
                    error=False,
                    metadata=item.metadata,
                    finish_reason=finish_reason,
                )
            )
        return results

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        if self.handle is None:
            self.start()
        if self._cli_binary is not None:
            return self._generate_via_llama_cli(batch)
        return chat_completions_batch(
            batch,
            client=self._client,
            model_name=self.handle.model_name,
            timeout=settings.SURYA_INFERENCE_TIMEOUT_SECONDS,
            max_workers=settings.SURYA_INFERENCE_PARALLEL or self.DEFAULT_PARALLEL,
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
            allow_stop_token_ids=False,  # vllm-only request extension
            # With --special the end token is part of the returned text, and llama-server's JSON-schema parser
            # rejects trailing text with a 500, so layout runs unconstrained there (the model writes valid JSON anyway).
            allow_guided_json=not settings.LLAMA_CPP_SPECIAL,
        )
