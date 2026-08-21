# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Surya is a document OCR/layout/table-recognition toolkit. Layout, OCR, and
table recognition all run through a single 650M-param VLM (Qwen3.5-style)
served via `vllm` (GPU) or `llama.cpp` (CPU/Apple Silicon). Text-line
detection and OCR-error detection are separate, smaller torch models that do
not need the VLM backend.

## Setup / commands

```bash
uv sync --group dev        # installs runtime + dev deps
uv run pytest              # run the full test suite
uv run pytest tests/test_layout.py::test_name   # run a single test
uv run surya_ocr DATA_PATH       # OCR CLI
uv run surya_layout DATA_PATH    # layout CLI
uv run surya_detect DATA_PATH    # text-line detection CLI
uv run surya_table DATA_PATH     # table recognition CLI
uv run surya_gui                 # streamlit interactive app
```

- Lint/format: `ruff` (config in `pyproject.toml`). `surya/common/rfdetr` is
  vendored byte-for-byte from upstream (Roboflow) and is excluded from
  linting/formatting — don't reformat it.
- Tests that need the VLM backend (`manager` fixture in `tests/conftest.py`)
  auto-skip if neither `vllm` nor `llama-server` is available in the
  environment, rather than failing.
- All settings live in `surya/settings.py` (pydantic-settings) and can be
  overridden via env vars or a `local.env` file — check there before adding
  new configuration.

## Architecture

### Inference manager (`surya/inference/`)

`SuryaInferenceManager` (`surya/inference/__init__.py`) is the single entry
point for VLM calls. One instance per process; construct it once and inject
it into predictors explicitly (`LayoutPredictor(manager)`,
`RecognitionPredictor(manager)`, `TableRecPredictor(manager)` all share the
same manager/server). It:

- Auto-selects a backend (`vllm` if an NVIDIA GPU is detected, else
  `llamacpp`) unless `SURYA_INFERENCE_BACKEND` is set explicitly.
- Lazily spawns the backend server on first `.start()` call (each CLI command
  spawns-and-tears-down by default; `--keep_server` /
  `SURYA_INFERENCE_KEEP_ALIVE` leaves it running for reuse across commands).
- Can attach to an already-running OpenAI-compatible server instead of
  spawning one, via `SURYA_INFERENCE_URL`.
- Backends live in `surya/inference/backends/` (`vllm.py`, `llamacpp.py`,
  `base.py` defines the `Backend` interface, `spawn.py` handles process
  spawning, `openai_client.py` is the shared OpenAI-compatible client).

Layout/OCR/table_rec predictors format prompts (`surya/inference/prompts.py`),
parse VLM responses (`surya/inference/parsers.py`), and validate against
schemas (`surya/inference/schema.py`).

### Predictors (per-task packages)

- `surya/detection/` — text-line detection (torch model, EfficientViT
  segformer, no VLM needed).
- `surya/layout/` — layout + reading order via the VLM. Canonical label
  set/relabeling in `surya/layout/label.py`.
- `surya/recognition/` — OCR (full-page or block-mode, VLM).
- `surya/table_rec/` — table row/column/cell recognition (VLM); simple mode
  (geometric intersections) or `predict_full` (HTML with spanning
  cells/headers).
- `surya/ocr_error/` — OCR-error detection (separate small torch model).
- `surya/fast_layout/` — lightweight CPU layout detector (rf-detr), an
  alternative to the VLM-based layout predictor for latency-sensitive use.

### Shared servers (`surya/common/batch_service/`)

Detection, OCR-error, and fast-layout each run as a shared server process (one
model instance serving all clients in a host) with continuous batching. First
client attaches-or-spawns; later clients just attach. `client.py`/`server.py`
implement this pattern; `config.py`/`serialize.py` handle request
serialization. Relevant settings (`*_SERVER_URL`, `*_SERVER_AUTOSTART`,
`*_SERVER_BATCH_WAIT_MS`, etc.) are per-service groups in
`surya/settings.py`.

### Output conventions

All predictors return polygons as
`[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]` with a derived axis-aligned `bbox`. OCR
output is HTML (`<math>...</math>` for equations, `<table>...</table>` for
tables) rather than plain text — there is no separate LaTeX-OCR pass; math is
handled inline by the same VLM call.

## Tests

`tests/conftest.py` provides session-scoped fixtures for each predictor,
built on shared `manager`/`detection_predictor`/`ocr_error_predictor`
fixtures. When adding tests for VLM-backed features, take the predictor
fixture (e.g. `layout_predictor`) rather than constructing a new
`SuryaInferenceManager`, so tests share one spawned server per session.
