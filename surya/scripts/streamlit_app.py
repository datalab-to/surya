"""Surya2 streamlit app — exercise layout, recognition, table_rec via the
inference manager. Detection + OCR-error stay in their own torch paths."""

from __future__ import annotations

import io
import re
import tempfile
import time
from typing import List

import pypdfium2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from surya.debug.draw import draw_polys_on_image, draw_bboxes_on_image
from surya.detection import TextDetectionResult
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import PageOCRResult
from surya.settings import settings
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult


# KaTeX-enabled HTML wrapper. The OCR HTML wraps math in <math>...</math>
# (KaTeX-compatible LaTeX inside), which a browser would otherwise show as
# raw text. We convert those tags to \( \) / \[ \] delimiters and let KaTeX
# auto-render typeset them inside an iframe component.
_KATEX_HEAD = r"""<!doctype html><html><head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
/* White "paper" card so the text stays readable in both light and dark
   Streamlit themes (the iframe is otherwise transparent and our text is dark). */
html,body{background:#ffffff;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.55;color:#111111;margin:0;padding:14px;}
table{border-collapse:collapse;margin:6px 0;} td,th{border:1px solid #bbb;padding:3px 6px;color:#111111;}
[data-label="SectionHeader"],[data-label="PageHeader"]{font-weight:600;}
</style></head><body>
"""

_KATEX_TAIL = r"""
<script>
renderMathInElement(document.body, {
  delimiters: [
    {left: "\\[", right: "\\]", display: true},
    {left: "\\(", right: "\\)", display: false}
  ],
  throwOnError: false
});
</script></body></html>
"""

_MATH_RE = re.compile(r"<math\b([^>]*)>(.*?)</math>", re.DOTALL | re.IGNORECASE)


def _math_to_katex(html_str: str) -> str:
    """Rewrite <math>...</math> tags into KaTeX \\( \\) / \\[ \\] delimiters."""

    def repl(m: "re.Match") -> str:
        attrs, inner = m.group(1), m.group(2)
        if re.search(r"""display\s*=\s*["']block["']""", attrs):
            return "\\[" + inner + "\\]"
        return "\\(" + inner + "\\)"

    return _MATH_RE.sub(repl, html_str or "")


def render_ocr_html(html_str: str, height: int = 400) -> None:
    """Render OCR HTML with math typeset by KaTeX (iframe component)."""
    components.html(
        _KATEX_HEAD + _math_to_katex(html_str) + _KATEX_TAIL,
        height=height,
        scrolling=True,
    )


def _assemble_page_html(page: PageOCRResult) -> str:
    """Reconstruct a div-block whole-page HTML from a PageOCRResult."""
    parts: List[str] = []
    for blk in page.blocks:
        if blk.skipped:
            continue
        x0, y0, x1, y1 = (int(c) for c in blk.bbox)
        body = blk.html or ""
        parts.append(
            f'<div data-bbox="{x0} {y0} {x1} {y1}" data-label="{blk.label}">{body}</div>'
        )
    return "\n".join(parts)


def _show_timing(label: str, elapsed_s: float, extra: str = "") -> None:
    """Render a small caption with wall-clock + optional extra detail."""
    detail = f" — {extra}" if extra else ""
    st.caption(f"⏱ {label}: {elapsed_s * 1000:.0f} ms ({elapsed_s:.2f}s){detail}")


@st.cache_resource()
def load_predictors_cached():
    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)
    rec_predictor = RecognitionPredictor(manager)
    table_rec_predictor = TableRecPredictor(manager)

    # Lazy-import detection / ocr_error to keep startup snappy when the user
    # only wants VLM modes
    from surya.detection import DetectionPredictor
    from surya.ocr_error import OCRErrorPredictor

    return {
        "manager": manager,
        "layout": layout_predictor,
        "recognition": rec_predictor,
        "table_rec": table_rec_predictor,
        "detection": DetectionPredictor(),
        "ocr_error": OCRErrorPredictor(),
    }


@st.cache_resource()
def load_fast_layout():
    from surya.fast_layout import FastLayoutPredictor

    return FastLayoutPredictor()


def _layout_predictor(use_fast: bool):
    return load_fast_layout() if use_fast else predictors["layout"]


def text_detection(img) -> tuple[Image.Image, TextDetectionResult, float]:
    t = time.perf_counter()
    text_pred = predictors["detection"]([img])[0]
    elapsed = time.perf_counter() - t
    text_polygons = [p.polygon for p in text_pred.bboxes]
    det_img = draw_polys_on_image(text_polygons, img.copy())
    return det_img, text_pred, elapsed


def layout_detection(
    img, use_fast: bool = False
) -> tuple[Image.Image, LayoutResult, float]:
    t = time.perf_counter()
    pred = _layout_predictor(use_fast)([img])[0]
    elapsed = time.perf_counter() - t
    polygons = [p.polygon for p in pred.bboxes]
    labels = [
        f"{p.label}-{p.position}-c{p.count}-{round(p.confidence or 0, 2)}"
        for p in pred.bboxes
    ]
    annotated = draw_polys_on_image(
        polygons, img.copy(), labels=labels, label_font_size=14
    )
    return annotated, pred, elapsed


def block_ocr(img) -> tuple[Image.Image, PageOCRResult, LayoutResult, float, float]:
    """Layout → block crops → BLOCK_PROMPT. Returns layout + block-OCR timings."""
    t_layout = time.perf_counter()
    layout = predictors["layout"]([img])[0]
    layout_elapsed = time.perf_counter() - t_layout

    t_blocks = time.perf_counter()
    page_results = predictors["recognition"]([img], [layout])
    blocks_elapsed = time.perf_counter() - t_blocks
    page = page_results[0]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)
    return annotated, page, layout, layout_elapsed, blocks_elapsed


def full_page_ocr(img) -> tuple[Image.Image, PageOCRResult, float]:
    """Single HIGH_ACCURACY_BBOX_PROMPT call on the whole page."""
    t = time.perf_counter()
    page_results = predictors["recognition"]([img], full_page=True)
    elapsed = time.perf_counter() - t
    page = page_results[0]
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)
    return annotated, page, elapsed


def table_recognition(
    img: Image.Image,
    mode: str,
    skip_table_detection: bool,
    use_fast_layout: bool = False,
) -> tuple[Image.Image, List[TableResult], float, float]:
    """Returns (annotated_img, table_preds, layout_elapsed, table_rec_elapsed)."""
    layout_elapsed = 0.0
    if skip_table_detection:
        table_imgs = [img]
        table_bboxes = [(0, 0, img.size[0], img.size[1])]
    else:
        t = time.perf_counter()
        layout = _layout_predictor(use_fast_layout)([img])[0]
        layout_elapsed = time.perf_counter() - t
        tables = [b for b in layout.bboxes if b.label in ("Table", "TableOfContents")]
        if not tables:
            return img.copy(), [], layout_elapsed, 0.0
        table_bboxes = [tuple(int(c) for c in b.bbox) for b in tables]
        table_imgs = [img.crop(b) for b in table_bboxes]

    t = time.perf_counter()
    if mode == "full":
        table_preds = predictors["table_rec"].predict_full(table_imgs)
    else:
        table_preds = predictors["table_rec"].predict_simple(table_imgs)
    table_rec_elapsed = time.perf_counter() - t

    out_img = img.copy()
    for pred, table_img, tbbox in zip(table_preds, table_imgs, table_bboxes):
        if pred.error or pred.mode != "simple" or not pred.rows:
            continue
        row_bboxes = [r.bbox for r in pred.rows]
        col_bboxes = [c.bbox for c in pred.cols]
        row_labels = [r.label for r in pred.rows]
        col_labels = [c.label for c in pred.cols]
        annot = table_img.copy()
        annot = draw_bboxes_on_image(
            row_bboxes, annot, labels=row_labels, label_font_size=14, color="blue"
        )
        annot = draw_bboxes_on_image(
            col_bboxes, annot, labels=col_labels, label_font_size=14, color="red"
        )
        # Paste annotated crop back at the table's position in the page.
        out_img.paste(annot, (tbbox[0], tbbox[1]))
    return out_img, table_preds, layout_elapsed, table_rec_elapsed


def ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    from pdftext.extraction import plain_text_output

    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        f.seek(0)

        page_middle = page_count // 2
        page_range = range(
            max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count)
        )
        text = plain_text_output(f.name, page_range=page_range)

    sample_gap = len(text) // max_samples
    if len(text) == 0 or sample_gap == 0:
        return "This PDF has no text or very little text", ["no text"]

    if sample_gap < sample_len:
        sample_gap = sample_len

    samples = []
    for i in range(0, len(text), sample_gap):
        samples.append(text[i : i + sample_len])

    results = predictors["ocr_error"](samples)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > 0.2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    page = doc.get_page(page_num - 1)
    bitmap = page.render(scale=dpi / 72)
    png = bitmap.to_pil()
    png_image = png.convert("RGB")
    doc.close()
    return png_image


@st.cache_data()
def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


st.set_page_config(layout="wide")
col1, col2 = st.columns([0.55, 0.45])

predictors = load_predictors_cached()

st.markdown(
    """
# Surya 2 Demo

VLM-backed layout, OCR, and table recognition. The model runs in a local
`llama-server` (or vllm) process, started on first use.

Modes:
- **Layout**: page → list of blocks with label + bbox + token count
- **Block OCR**: layout + per-block HTML
- **Table Rec (simple)**: row + column bboxes only
- **Table Rec (full)**: full HTML for each detected table
"""
)

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.stop()

filetype = in_file.type
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_number = st.sidebar.number_input(
        f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count
    )
    # Render at high DPI so the OCR / table-rec demos see fine glyphs.
    # Layout + detection internally downsample (or accept the small perf hit
    # at demo scale); we always render and display the high-DPI page here.
    pil_image = get_page_image(in_file, page_number, settings.IMAGE_DPI_HIGHRES)
else:
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

run_full_page_ocr = st.sidebar.button("Run Full-Page OCR")
run_text_det = st.sidebar.button("Run Text Detection")
run_layout = st.sidebar.button("Run Layout Analysis")
run_table_rec = st.sidebar.button("Run Table Rec")
run_block_ocr = st.sidebar.button("Run Block OCR")
run_ocr_errors = st.sidebar.button("Run bad-PDF-text detection")

use_fast_layout = st.sidebar.checkbox(
    "Fast layout",
    value=True,
    help="Use the fast layout detector.",
)
table_mode = st.sidebar.radio(
    "Table mode",
    options=["simple", "full"],
    index=0,
    help="simple: rows+cols only. full: full HTML.",
)
skip_table_detection = st.sidebar.checkbox(
    "Skip table detection",
    value=False,
    help="Treat the entire page/image as a single table.",
)

if pil_image is None:
    st.stop()


if run_text_det:
    det_img, text_pred, elapsed = text_detection(pil_image)
    with col1:
        _show_timing("Text detection", elapsed, f"{len(text_pred.bboxes)} polys")
        st.image(det_img, caption="Detected Text", use_container_width=True)
        st.json(
            text_pred.model_dump(exclude=["heatmap", "affinity_map"]), expanded=False
        )


if run_layout:
    annotated, pred, elapsed = layout_detection(pil_image, use_fast=use_fast_layout)
    with col1:
        label = "Layout (fast)" if use_fast_layout else "Layout"
        _show_timing(label, elapsed, f"{len(pred.bboxes)} blocks")
        st.image(annotated, caption="Detected Layout", use_container_width=True)
        st.json(pred.model_dump(), expanded=False)


if run_block_ocr:
    annotated, page, layout, t_layout, t_blocks = block_ocr(pil_image)
    with col1:
        n_blocks = len(page.blocks)
        n_ok = sum(1 for b in page.blocks if not b.skipped and not b.error)
        _show_timing("Block OCR — layout", t_layout, f"{n_blocks} blocks")
        _show_timing("Block OCR — per-block OCR", t_blocks, f"{n_ok} OCR'd")
        _show_timing("Block OCR — total", t_layout + t_blocks)
        st.image(
            annotated,
            caption="Block OCR (green=ok, orange=skipped, red=error)",
            use_container_width=True,
        )
        full_html = _assemble_page_html(page)
        with st.expander("Full page HTML (rendered)", expanded=False):
            render_ocr_html(full_html, height=600)
        with st.expander("Full page HTML (source)", expanded=False):
            st.code(full_html, language="html")
        for blk in page.blocks:
            with st.expander(
                f"#{blk.reading_order} {blk.label} (conf {blk.confidence:.2f})"
            ):
                # Diagnostics: show numeric bbox + polygon + a thumbnail with the
                # drawn rectangle highlighted, then the actual crop fed to OCR.
                xs = [p[0] for p in blk.polygon]
                ys = [p[1] for p in blk.polygon]
                bbox_drawn = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                cx0 = max(0, int(min(xs)) - 4)
                cy0 = max(0, int(min(ys)) - 4)
                cx1 = min(pil_image.size[0], int(max(xs)) + 4)
                cy1 = min(pil_image.size[1], int(max(ys)) + 4)
                st.text(
                    f"bbox(drawn) = {bbox_drawn}\n"
                    f"crop(ocr)  = {(cx0, cy0, cx1, cy1)}  (= bbox ± 4px pad)"
                )
                # Thumbnail with this block's rectangle highlighted in red.
                thumb = pil_image.copy()
                ImageDraw.Draw(thumb).rectangle(bbox_drawn, outline="red", width=4)
                st.image(thumb, caption="this block's drawn rect (red)", width=300)
                # The actual crop fed to OCR
                if cx1 > cx0 and cy1 > cy0:
                    st.image(pil_image.crop((cx0, cy0, cx1, cy1)), caption="OCR crop")
                if blk.skipped:
                    st.info("Block skipped (visual label)")
                elif blk.error:
                    st.error("Block OCR errored")
                else:
                    render_ocr_html(blk.html, height=160)
                    st.code(blk.html, language="html")


if run_full_page_ocr:
    annotated, page, elapsed = full_page_ocr(pil_image)
    with col1:
        n_blocks = len(page.blocks)
        n_ok = sum(1 for b in page.blocks if not b.skipped and not b.error)
        _show_timing("Full-Page OCR", elapsed, f"{n_blocks} blocks parsed, {n_ok} OK")
        st.image(
            annotated,
            caption="Full-Page OCR (green=ok, orange=skipped, red=error)",
            use_container_width=True,
        )
        full_html = _assemble_page_html(page)
        with st.expander("Full page HTML (rendered)", expanded=False):
            render_ocr_html(full_html, height=600)
        with st.expander("Full page HTML (source)", expanded=False):
            st.code(full_html, language="html")
        for blk in page.blocks:
            with st.expander(
                f"#{blk.reading_order} {blk.label} (conf {blk.confidence:.2f})"
            ):
                if blk.skipped:
                    st.info("Block skipped (visual label)")
                elif blk.error:
                    st.error("Block OCR errored")
                else:
                    render_ocr_html(blk.html, height=160)
                    st.code(blk.html, language="html")


if run_table_rec:
    table_img, preds, t_layout, t_table = table_recognition(
        pil_image, table_mode, skip_table_detection, use_fast_layout=use_fast_layout
    )
    with col1:
        if not skip_table_detection:
            _show_timing("Table Rec — layout", t_layout, f"{len(preds)} tables found")
        _show_timing(f"Table Rec — {table_mode}", t_table)
        if not skip_table_detection:
            _show_timing("Table Rec — total", t_layout + t_table)
        st.image(table_img, caption="Table Recognition", use_container_width=True)
        for pred in preds:
            if pred.mode == "full" and pred.html:
                with st.expander("Table HTML"):
                    render_ocr_html(pred.html, height=400)
                    st.code(pred.html, language="html")
            else:
                st.json(pred.model_dump(), expanded=False)


if run_ocr_errors:
    if "pdf" not in filetype:
        st.error("This feature only works with PDFs.")
    else:
        label, results = ocr_errors(in_file, page_count)
        with col1:
            st.write(label)
            st.json(results)


with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
