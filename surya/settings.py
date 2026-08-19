import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
from pathlib import Path
from platformdirs import user_cache_dir


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96  # used for layout + text detection (coarse structure)
    IMAGE_DPI_HIGHRES: int = 192  # used for recognition + table rec (fine glyphs)
    IN_STREAMLIT: bool = False
    DISABLE_TQDM: bool = False
    S3_BASE_URL: str = "https://models.datalab.to"
    PARALLEL_DOWNLOAD_WORKERS: int = 10
    MODEL_CACHE_DIR: str = str(Path(user_cache_dir("datalab")) / "models")
    LOGLEVEL: str = "INFO"

    # Paths
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ---- Surya2 inference (VLM-backed: vllm | llamacpp) ---------------------
    SURYA_MODEL_CHECKPOINT: str = "datalab-to/surya-ocr-2"
    SURYA_GGUF_REPO: str = "datalab-to/surya-ocr-2-gguf"
    SURYA_GGUF_MODEL_FILE: str = "surya-2.gguf"
    SURYA_GGUF_MMPROJ_FILE: str = "surya-2-mmproj.gguf"
    # If set, used directly instead of HF download (handy for local-conversion testing)
    SURYA_GGUF_LOCAL_MODEL_PATH: Optional[str] = None
    SURYA_GGUF_LOCAL_MMPROJ_PATH: Optional[str] = None

    # Backend selection
    SURYA_INFERENCE_BACKEND: Optional[str] = None  # "vllm" | "llamacpp" | None (auto)
    SURYA_INFERENCE_URL: Optional[str] = None  # external server, skip spawn
    SURYA_INFERENCE_AUTOSTART: bool = True
    # Leave an auto-spawned server running after the process exits so later
    # commands attach to it instead of re-spawning (avoids repeated startup /
    # model-load cost). Stop it manually when done — see `surya/inference`.
    SURYA_INFERENCE_KEEP_ALIVE: bool = False
    SURYA_INFERENCE_HOST: str = "127.0.0.1"
    SURYA_INFERENCE_PORT: Optional[int] = None  # None = pick a free port
    # Client-side concurrent request count. None = let the backend pick a
    # sensible default (vllm scales to the server's max_num_seqs / GPU
    # capacity; llama.cpp uses a conservative slot count). Set an int to
    # override.
    SURYA_INFERENCE_PARALLEL: Optional[int] = None
    # Per-parallel-slot KV-cache budget for the llama.cpp backend. Worst-case
    # one OCR request: ~2k for image prefill + SURYA_MAX_TOKENS_FULL_PAGE
    # (8192) generation + ~2k prompt/chat-template overhead ≈ 12k. Below this
    # llama-server silently truncates outputs once a slot fills.
    SURYA_INFERENCE_CTX_PER_SLOT: int = 12288
    # Optional override for the *total* ctx passed to llama-server. When None
    # (default), total = max(16384, PARALLEL * CTX_PER_SLOT). Set this only
    # if you've hand-tuned for a specific machine.
    SURYA_INFERENCE_CTX_SIZE: Optional[int] = None
    SURYA_INFERENCE_TIMEOUT_SECONDS: float = 600.0
    SURYA_INFERENCE_STARTUP_TIMEOUT: float = 600.0
    SURYA_INFERENCE_LOGPROBS: bool = True
    # Force layout/table_rec output through a JSON schema via guided decoding.
    # Eliminates malformed-JSON failures at small decode-throughput cost.
    SURYA_GUIDED_LAYOUT: bool = True
    # Disabled: with no minItems in TABLE_REC_JSON_SCHEMA, the constrained
    # decoder closes the array after one element at temperature=0. The model
    # produces well-formed JSON without the schema.
    SURYA_GUIDED_TABLE_REC: bool = False

    # Token budgets
    SURYA_MAX_TOKENS_LAYOUT: int = 3072
    SURYA_MAX_TOKENS_TABLE_REC: int = 3072
    SURYA_MAX_TOKENS_BLOCK_CEILING: int = 8192
    SURYA_MAX_TOKENS_FULL_PAGE: int = 12288

    # Full-page OCR: progressive-temperature regeneration before block-mode fallback.
    # Off by default (single greedy pass -> block fallback); opt-in for benchmarking.
    SURYA_FULLPAGE_REGEN: bool = False

    BBOX_SCALE: int = 1000

    # vllm
    VLLM_DOCKER_IMAGE: str = "docker.io/vllm/vllm-openai:v0.20.1"
    VLLM_API_KEY: str = "EMPTY"
    VLLM_GPUS: str = "0"
    VLLM_GPU_TYPE: str = "4090"
    # bfloat16 needs an Ampere+ GPU (compute capability >= 8.0). On older cards
    # (e.g. T4 / Turing) vllm refuses to start with bf16 — set float16 there.
    VLLM_DTYPE: str = "bfloat16"
    VLLM_MAX_MODEL_LEN: int = 18000
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.85
    VLLM_ENABLE_MTP: bool = True
    VLLM_MTP_TOKENS: int = 2
    VLLM_EXTRA_ARGS: Optional[str] = None
    DOCKER_HF_CACHE_PATH: str = "~/.cache/huggingface"

    # llama.cpp
    LLAMA_CPP_BINARY: str = "llama-server"
    LLAMA_CPP_NGL: int = 99  # all layers on GPU (Metal on macOS, CUDA on Linux GPU); harmless no-op on pure-CPU builds
    LLAMA_CPP_NO_MMPROJ_OFFLOAD: bool = False
    LLAMA_CPP_EXTRA_ARGS: Optional[str] = None

    # ---- Detection (kept) ---------------------------------------------------
    DETECTOR_BATCH_SIZE: Optional[int] = None
    DETECTOR_MODEL_CHECKPOINT: str = "s3://text_detection/2025_05_07"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1400
    DETECTOR_TEXT_THRESHOLD: float = 0.6
    DETECTOR_BLANK_THRESHOLD: float = 0.35
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(8, os.cpu_count())
    DETECTOR_MIN_PARALLEL_THRESH: int = 3
    DETECTOR_BOX_Y_EXPAND_MARGIN: float = 0.05
    # Shared detection server (one model instance for all clients; see the
    # fast-layout block below for what each knob does).
    DETECTOR_SERVER_URL: Optional[str] = None
    DETECTOR_SERVER_HOST: str = "127.0.0.1"
    DETECTOR_SERVER_PORT: Optional[int] = None
    DETECTOR_SERVER_AUTOSTART: bool = True
    DETECTOR_SERVER_STARTUP_TIMEOUT: float = 300.0
    DETECTOR_SERVER_TIMEOUT: float = 600.0
    DETECTOR_SERVER_BATCH_WAIT_MS: int = 5
    DETECTOR_SERVER_MAX_BATCH: Optional[int] = None

    # ---- OCR Error (kept) ---------------------------------------------------
    OCR_ERROR_MODEL_CHECKPOINT: str = "s3://ocr_error_detection/2025_02_18"
    OCR_ERROR_BATCH_SIZE: Optional[int] = None
    # Shared ocr-error server (one model instance for all clients).
    OCR_ERROR_SERVER_URL: Optional[str] = None
    OCR_ERROR_SERVER_HOST: str = "127.0.0.1"
    OCR_ERROR_SERVER_PORT: Optional[int] = None
    OCR_ERROR_SERVER_AUTOSTART: bool = True
    OCR_ERROR_SERVER_STARTUP_TIMEOUT: float = 300.0
    OCR_ERROR_SERVER_TIMEOUT: float = 600.0
    OCR_ERROR_SERVER_BATCH_WAIT_MS: int = 5
    OCR_ERROR_SERVER_MAX_BATCH: Optional[int] = None

    # ---- Fast layout (rf-detr, CPU) ------------------------------------------
    # Lightweight detector. Checkpoint may be a local dir (rf-detr .pth + config.json),
    # an hf://<repo>/<subfolder> ref, or an s3:// path.
    # Override via FAST_LAYOUT_MODEL_CHECKPOINT.
    FAST_LAYOUT_MODEL_CHECKPOINT: str = "hf://datalab-to/surya_layout2"
    FAST_LAYOUT_BATCH_SIZE: Optional[int] = None
    FAST_LAYOUT_CONFIDENCE_THRESHOLD: float = 0.4
    # Post-process: if a same-label block is contained within (or overlaps almost
    # entirely with) a larger same-label block, drop the smaller and merge its
    # extent into the larger. This is the fraction of the smaller box that must lie
    # inside the larger to trigger the merge. Set to >1 to disable.
    FAST_LAYOUT_CONTAINMENT_THRESHOLD: float = 0.9
    FAST_ORDER_MODEL_CHECKPOINT: str = "hf://datalab-to/surya_layout2/order"
    # Run the learned reading-order head after fast layout. When False, boxes come
    # back in raster order (top-to-bottom, left-to-right) and the order model is
    # neither loaded nor run — saves latency at the cost of reading-order quality.
    FAST_LAYOUT_USE_ORDER: bool = True
    # Device for the rf-detr fast detector. None = auto (cuda > mps > cpu). Override to
    # force "cpu"/"cuda"/"mps".
    FAST_DETECTOR_DEVICE: Optional[str] = None
    # torch intra-op thread count for the rf-detr detector, applied inside the shared
    # server. None = torch default (all cores).
    FAST_LAYOUT_NUM_THREADS: Optional[int] = None

    # ---- Fast layout server (shared, continuous-batching) --------------------
    # The rf-detr model always runs in one shared server process; FastLayoutPredictor
    # is a thin client of it (there's no benefit to more than one layout model on a
    # host, and N in-process copies thrash the CPU/GPU). The first client attaches to
    # a running server or spawns one; the rest attach. These settings tune that server.
    # Pin an already-running server; skips spawn. e.g. "http://127.0.0.1:8920/v1".
    FAST_LAYOUT_SERVER_URL: Optional[str] = None
    FAST_LAYOUT_SERVER_HOST: str = "127.0.0.1"
    FAST_LAYOUT_SERVER_PORT: Optional[int] = None  # None = pick a free port
    FAST_LAYOUT_SERVER_AUTOSTART: bool = True
    FAST_LAYOUT_SERVER_STARTUP_TIMEOUT: float = 300.0
    FAST_LAYOUT_SERVER_TIMEOUT: float = 600.0  # client request timeout
    # Server-side continuous batching: coalesce pages arriving within this window
    # (across all client requests) into one detect() call. Larger = better batch
    # efficiency (esp. on GPU), slightly more latency.
    FAST_LAYOUT_SERVER_BATCH_WAIT_MS: int = 5
    # Max pages per coalesced detect() call. None = fall back to FAST_LAYOUT_BATCH_SIZE
    # (or 8). This is the server's batch ceiling regardless of per-request sizes.
    FAST_LAYOUT_SERVER_MAX_BATCH: Optional[int] = None

    # ---- Debug / draw fonts (label rendering on annotated images) ----------
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }
    RECOGNITION_FONT_DL_BASE: str = (
        "https://github.com/satbyy/go-noto-universal/releases/download/v7.0"
    )

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.float16

    @computed_field
    def MODEL_DTYPE_BFLOAT(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.bfloat16

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
