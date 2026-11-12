import os
from typing import Dict, Optional

from dotenv import find_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    IMAGE_DPI: int = 96  # used for layout detection (coarse structure)
    IMAGE_DPI_HIGHRES: int = 192  # used for recognition + table rec (fine glyphs)
    DISABLE_TQDM: bool = False
    LOGLEVEL: str = "INFO"

    # Paths
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    # ---- Surya VLM inference (vllm | llamacpp | mlx) ------------------------
    SURYA_MODEL_CHECKPOINT: str = "datalab-to/surya-ocr-2.5"
    SURYA_GGUF_REPO: str = "datalab-to/surya-ocr-2.5-gguf"
    SURYA_GGUF_MODEL_FILE: str = "surya-2.5.gguf"
    SURYA_GGUF_MMPROJ_FILE: str = "surya-2.5-mmproj.gguf"
    # If set, used directly instead of HF download (handy for local-conversion testing)
    SURYA_GGUF_LOCAL_MODEL_PATH: Optional[str] = None
    SURYA_GGUF_LOCAL_MMPROJ_PATH: Optional[str] = None

    # Backend selection
    SURYA_INFERENCE_BACKEND: Optional[str] = (
        None  # "vllm" | "llamacpp" | "mlx" | None (auto: vllm on NVIDIA, else llamacpp)
    )
    SURYA_INFERENCE_URL: Optional[str] = None  # external server, skip spawn
    SURYA_INFERENCE_API_KEY: Optional[str] = None
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
    # Per-parallel-slot KV-cache budget for the llama.cpp backend.
    SURYA_INFERENCE_CTX_PER_SLOT: int = 12288
    # Optional override for the *total* ctx passed to llama-server. When None
    # (default), total = max(16384, PARALLEL * CTX_PER_SLOT).
    SURYA_INFERENCE_CTX_SIZE: Optional[int] = None
    SURYA_INFERENCE_TIMEOUT_SECONDS: float = 600.0
    SURYA_INFERENCE_STARTUP_TIMEOUT: float = 600.0
    SURYA_INFERENCE_LOGPROBS: bool = True
    # Force layout output through a JSON schema via guided decoding.
    SURYA_GUIDED_LAYOUT: bool = True

    # Token budgets
    SURYA_MAX_TOKENS_LAYOUT: int = 3072
    SURYA_MAX_TOKENS_BLOCK_CEILING: int = 8192
    SURYA_MAX_TOKENS_FULL_PAGE: int = 12288

    # Optional stop handling forwarded on every request.
    SURYA_STOP_STRINGS: Optional[str] = None  # comma-separated
    SURYA_STOP_TOKEN_IDS: Optional[str] = None  # comma-separated ints (vllm only)

    # Full-page OCR: progressive-temperature regeneration before block-mode fallback.
    # Off by default (single greedy pass -> block fallback); opt-in for benchmarking.
    SURYA_FULLPAGE_REGEN: bool = False

    BBOX_SCALE: int = 1000

    # vllm
    VLLM_DOCKER_IMAGE: str = "vllm/vllm-openai:v0.25.1"
    # Container runtime used to run VLLM_DOCKER_IMAGE. None = auto: `docker` if it is on
    # PATH, otherwise `podman`. Set to "docker", "podman" or a path to either binary.
    SURYA_CONTAINER_RUNTIME: Optional[str] = None
    VLLM_API_KEY: Optional[str] = None  # legacy alias for SURYA_INFERENCE_API_KEY
    VLLM_GPUS: str = "0"
    VLLM_GPU_TYPE: str = "4090"
    VLLM_DTYPE: str = "bfloat16"
    VLLM_MAX_MODEL_LEN: int = 24576
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.85
    # None = auto: enable MTP speculation only when the checkpoint ships an MTP head.
    VLLM_ENABLE_MTP: Optional[bool] = None
    VLLM_MTP_TOKENS: int = 2
    VLLM_EXTRA_ARGS: Optional[str] = None
    DOCKER_HF_CACHE_PATH: str = "~/.cache/huggingface"

    # llama.cpp
    LLAMA_CPP_BINARY: str = "llama-server"
    LLAMA_CPP_NGL: int = 99
    LLAMA_CPP_NO_MMPROJ_OFFLOAD: bool = False
    LLAMA_CPP_SPECIAL: bool = True
    LLAMA_CPP_EXTRA_ARGS: Optional[str] = None

    # mlx (Apple Silicon): SURYA_INFERENCE_BACKEND=mlx spawns `mlx_vlm.server`
    SURYA_MLX_MODEL: Optional[str] = None
    MLX_VLM_EXTRA_ARGS: Optional[str] = None

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

    @model_validator(mode="after")
    def _resolve_api_key(self):
        resolved = self.SURYA_INFERENCE_API_KEY or self.VLLM_API_KEY or "EMPTY"
        self.SURYA_INFERENCE_API_KEY = resolved
        self.VLLM_API_KEY = resolved
        return self

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
