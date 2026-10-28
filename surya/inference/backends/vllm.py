"""vllm backend: spawns the vllm/vllm-openai docker image with MTP=2."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from typing import List, Optional

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


# 24GB baseline (re-tune for surya-2 once benchmarks land)
BASELINE_VRAM_GB = 24
BASELINE_MAX_BATCHED_TOKENS = 8192
BASELINE_MAX_NUM_SEQS = 32

GPU_VRAM_GB = {
    "b300": 270,
    "b200": 180,
    "h200": 141,
    "h100": 80,
    "a100-80": 80,
    "a100": 40,
    "a100-40": 40,
    "l40s": 48,
    "a10": 24,
    "l4": 24,
    "5090": 32,
    "4090": 24,
    "3090": 24,
    "t4": 16,
}


def _gpu_settings(gpu: str) -> tuple[int, int]:
    vram = GPU_VRAM_GB.get(gpu)
    if vram is None:
        available = ", ".join(sorted(GPU_VRAM_GB.keys()))
        raise SpawnError(f"Unknown VLLM_GPU_TYPE {gpu!r}. Available: {available}")
    ratio = vram / BASELINE_VRAM_GB
    raw_tokens = BASELINE_MAX_BATCHED_TOKENS * ratio
    max_batched_tokens = max(1024, 2 ** math.floor(math.log2(raw_tokens)))
    max_num_seqs = max(8, (int(BASELINE_MAX_NUM_SEQS * ratio) // 8) * 8)
    return max_batched_tokens, max_num_seqs


def _resolve_docker_binary() -> str:
    found = shutil.which("docker")
    if found:
        return found
    raise SpawnError(
        "docker binary not found. Install Docker (https://docs.docker.com/get-docker/) "
        "and ensure the daemon is running."
    )


def _health_url(port: int) -> str:
    return f"http://{settings.SURYA_INFERENCE_HOST}:{port}"


def _openai_url(port: int) -> str:
    return f"http://{settings.SURYA_INFERENCE_HOST}:{port}/v1"


class VllmBackend(Backend):
    name = "vllm"

    # Cap auto-scaled client concurrency at the GPU-saturation knee. Measured
    # layout throughput on a B200 (max_num_seqs=240): 48→96 gives +28%, but
    # 96→240 only +5% — the GPU is compute-bound past ~96 concurrent, so extra
    # requests just queue while adding thread/connection overhead. Smaller GPUs
    # are bounded by their own (lower) max_num_seqs below this.
    MAX_AUTO_PARALLEL = 96

    def __init__(self):
        self.handle: Optional[ServerHandle] = None
        self._client: Optional[OpenAI] = None
        # Server concurrency capacity, set when the server is configured;
        # used to default client-side parallelism to the GPU's capability.
        self._max_num_seqs: int = 8

    def capacity(self) -> int:
        return self._max_num_seqs

    def _client_parallel(self) -> int:
        if settings.SURYA_INFERENCE_PARALLEL is not None:
            return settings.SURYA_INFERENCE_PARALLEL
        return min(self._max_num_seqs, self.MAX_AUTO_PARALLEL)

    def start(self) -> ServerHandle:
        if self.handle is not None:
            return self.handle

        # Best-effort server capacity for defaulting client concurrency, even
        # when attaching to an external server.
        try:
            self._max_num_seqs = _gpu_settings(settings.VLLM_GPU_TYPE)[1]
        except SpawnError:
            pass

        # If user pinned an external server, attach without spawning docker.
        if settings.SURYA_INFERENCE_URL:
            spawned = attach_or_spawn(
                backend=self.name,
                expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
                spawn_fn=lambda port: SpawnHandle(
                    pid=None, cleanup_id="", cleanup_kind="docker"
                ),
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
                api_key=settings.VLLM_API_KEY, base_url=self.handle.base_url
            )
            return self.handle

        docker = _resolve_docker_binary()
        max_batched_tokens, max_num_seqs = _gpu_settings(settings.VLLM_GPU_TYPE)
        self._max_num_seqs = max_num_seqs

        def spawn_fn(port: int) -> SpawnHandle:
            container_name = f"surya-vllm-{port}"
            hf_cache = os.path.expanduser(settings.DOCKER_HF_CACHE_PATH)
            cmd = [
                docker,
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                "--runtime",
                "nvidia",
                "--gpus",
                f"device={settings.VLLM_GPUS}",
                "-v",
                f"{hf_cache}:/root/.cache/huggingface",
                "-p",
                f"{settings.SURYA_INFERENCE_HOST}:{port}:8000",
                "--ipc=host",
                settings.VLLM_DOCKER_IMAGE,
                "--model",
                settings.SURYA_MODEL_CHECKPOINT,
                "--no-enforce-eager",
                "--max-num-seqs",
                str(max_num_seqs),
                "--dtype",
                settings.VLLM_DTYPE,
                "--max-model-len",
                str(settings.VLLM_MAX_MODEL_LEN),
                "--max-num-batched-tokens",
                str(max_batched_tokens),
                "--gpu-memory-utilization",
                str(settings.VLLM_GPU_MEMORY_UTILIZATION),
                "--enable-prefix-caching",
                "--mm-processor-kwargs",
                json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
                "--served-model-name",
                settings.SURYA_MODEL_CHECKPOINT,
            ]
            if settings.VLLM_ENABLE_MTP:
                spec_config = json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": settings.VLLM_MTP_TOKENS,
                    }
                )
                cmd.extend(["--speculative-config", spec_config])
            for extra in (settings.VLLM_EXTRA_ARGS or "").split():
                cmd.append(extra)
            logger.info(f"Spawning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise SpawnError(f"docker run failed: {result.stderr or result.stdout}")
            return SpawnHandle(
                pid=None, cleanup_id=container_name, cleanup_kind="docker"
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
            api_key=settings.VLLM_API_KEY,
            base_url=self.handle.base_url,
        )
        return self.handle

    def stop(self) -> None:
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
            max_workers=self._client_parallel(),
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
        )
