---
name: surya
description: >-
  Run document OCR, layout analysis, reading order detection, and HTML table
  recognition in 90+ languages using Surya 2.5 (surya-ocr). Use this skill
  whenever the user or task needs to OCR PDFs or images, detect document layout
  boxes and reading order, extract HTML tables with colspan/rowspan, or launch
  the interactive Surya web viewer.
---

# Surya 2.5 Document OCR & Layout Skill (`surya-ocr`)

Surya 2.5 is a 689M-parameter document vision-language model (VLM) for **full-page and per-block OCR** (HTML + KaTeX LaTeX `<math>`), **layout analysis & reading order**, and **HTML table recognition** (`<table>` with `colspan` / `rowspan`) across 90+ languages.

---

## 1. Modes & Capabilities

| Task | CLI Command | Python Predictor | Description |
| :--- | :--- | :--- | :--- |
| **Full-Page OCR (default)** | `surya_ocr` | `RecognitionPredictor(full_page=True)` | Single VLM call per page returning layout bboxes, labels, reading order, and HTML content (with automatic block-mode fallback on truncation/loops). |
| **Block OCR** | `surya_ocr --block_mode` | `RecognitionPredictor(full_page=False)` | Two-stage pipeline: `LayoutPredictor` (96 DPI) followed by per-block crop OCR (192 DPI). |
| **Layout & Reading Order** | `surya_layout` | `LayoutPredictor` | Detects bounding boxes, canonical labels (`Text`, `SectionHeader`, `Table`, `Equation`, `Picture`, `Form`, etc.), reading order, and token-count estimates. |
| **Table Recognition** | `surya_table` | `TableRecPredictor` | Extracts tables as full HTML (`<table>`, `<tr>`, `<th>`, `<td>`, `colspan`, `rowspan`). |
| **Interactive Viewer** | `surya_gui` | `surya.scripts.viewer` | Browser UI to inspect layout boxes, reading order, rendered HTML/math, and JSON page by page. |

---

## 2. CLI Quick Reference

### OCR (`surya_ocr`)

```bash
# Print OCR JSON directly to stdout (best for AI agents & shell pipelines)
surya_ocr document.pdf --stdout

# OCR specific pages (0-indexed) to stdout
surya_ocr document.pdf --page_range "0,2-5" --stdout

# Save results.json and annotated block images to a directory
surya_ocr document.pdf --output_dir ./results --images

# Run two-stage layout + per-block crop OCR instead of full-page OCR
surya_ocr document.pdf --block_mode --output_dir ./results

# Recursively OCR a folder of PDFs/images and keep the VLM server alive for later commands
surya_ocr ./input_folder --recursive --keep_server --output_dir ./results
```

### Layout & Reading Order (`surya_layout`)

```bash
# Detect layout boxes and print JSON to stdout
surya_layout document.pdf --stdout

# Save layout JSON and annotated layout images
surya_layout document.pdf --images --output_dir ./results
```

### Table Recognition (`surya_table`)

```bash
# Detect tables on pages and extract HTML tables to stdout
surya_table document.pdf --stdout

# Recognize an already-cropped table image directly
surya_table table_crop.png --skip_table_detection --stdout
```

### Interactive Web Viewer (`surya_gui`)

```bash
# Launch the self-contained interactive web viewer
surya_gui --port 8000
```

---

## 3. Python API Recipes

```python
from PIL import Image
from surya import (
    LayoutPredictor,
    RecognitionPredictor,
    SuryaInferenceManager,
    TableRecPredictor,
)

manager = SuryaInferenceManager()  # auto-spawns vllm (GPU) or llama-server (CPU/Metal)
image = Image.open("page.png").convert("RGB")

# 1. Full-page OCR (default: one VLM call per page)
rec_predictor = RecognitionPredictor(manager)
page_results = rec_predictor([image])
for blk in page_results[0].blocks:
    print(blk.reading_order, blk.label, blk.bbox, blk.html)

# Optional: also OCR text inside Figure / Picture / Diagram regions
page_results_all = rec_predictor([image], skip_labels=[])

# 2. Layout detection + Block-mode OCR
layout_predictor = LayoutPredictor(manager)
layouts = layout_predictor([image])
block_results = rec_predictor([image], layouts)

# 3. HTML Table Recognition (on cropped table images)
table_predictor = TableRecPredictor(manager)
table_results = table_predictor([image])
print(table_results[0].html)
```
