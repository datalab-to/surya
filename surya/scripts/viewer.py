"""Interactive web viewer for Surya 2.5 (modelled on marker_gui / indra-ocr-viewer).

Run standalone or via `surya_gui`:
    surya_gui --port 8000
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import re
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.parse
import uuid

from bs4 import BeautifulSoup
import click
from PIL import Image
import pypdfium2

from surya.common.util import expand_bbox
from surya.inference import SuryaInferenceManager
from surya.inference.parsers import sanitize_ocr_html
from surya.layout import LayoutPredictor
from surya.logging import get_logger
from surya.recognition import RecognitionPredictor
from surya.settings import settings
from surya.table_rec import TableRecPredictor

logger = get_logger()

SURYA_VERSION = "0.24.0"
MAX_SIDE = 1600
MODES = ("full_page", "block_ocr", "layout", "table_rec")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
}


def _load_logo_b64() -> str:
    logo_path = os.path.join(settings.BASE_DIR, "static", "datalab-logo.png")
    if os.path.isfile(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except OSError:
            pass
    return ""


LOGO = _load_logo_b64()


def _strip_html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    return " ".join(soup.get_text(" ").split())


def _scale_bbox_rect(
    bbox: List[float], src_size: tuple[int, int], dst_size: tuple[int, int]
) -> List[float]:
    sw, sh = max(1.0, float(src_size[0])), max(1.0, float(src_size[1]))
    dw, dh = float(dst_size[0]), float(dst_size[1])
    x0, y0, x1, y1 = bbox[:4]
    return [
        round(x0 / sw * dw, 1),
        round(y0 / sh * dh, 1),
        round(x1 / sw * dw, 1),
        round(y1 / sh * dh, 1),
    ]


class ViewerEngine:
    """Holds uploaded documents and processes one page per request with the
    selected Surya mode so the browser displays each page as it finishes."""

    def __init__(
        self,
        predictor_supplier: Optional[Callable[[], Dict[str, Any]]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        max_upload_bytes: int = 100 * 1024 * 1024,
    ):
        self._predictor_supplier = predictor_supplier
        self._predictors: Optional[Dict[str, Any]] = None
        self.max_upload_bytes = max_upload_bytes
        self.defaults = defaults or {
            "mode": "full_page",
            "skip_table_detection": False,
            "include_figures": False,
            "dpi": settings.IMAGE_DPI_HIGHRES,
            "max_pages": 20,
        }
        self.tmp = tempfile.TemporaryDirectory(prefix="surya_viewer_")
        self.docs_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.docs: Dict[str, Dict[str, Any]] = {}

    def get_predictors(self) -> Dict[str, Any]:
        if self._predictor_supplier is not None:
            preds = self._predictor_supplier()
            if preds is not None:
                return preds
        if self._predictors is None:
            with self.model_lock:
                if self._predictors is None:
                    manager = SuryaInferenceManager()
                    self._predictors = {
                        "manager": manager,
                        "layout": LayoutPredictor(manager),
                        "recognition": RecognitionPredictor(manager),
                        "table_rec": TableRecPredictor(manager),
                    }
        return self._predictors

    def info(self) -> Dict[str, Any]:
        return {
            "version": SURYA_VERSION,
            "model": settings.SURYA_MODEL_CHECKPOINT,
            "modes": list(MODES),
            "defaults": self.defaults,
        }

    def upload(self, data: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        if not data:
            raise ValueError("Uploaded file is empty.")
        if len(data) > self.max_upload_bytes:
            raise ValueError("Uploaded file exceeds maximum allowed size.")

        raw_name = os.path.basename(filename or "document.pdf")
        ext = os.path.splitext(raw_name)[1].lower()
        if not ext:
            ext = ".pdf" if data[:5] == b"%PDF-" else ".png"
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        did = uuid.uuid4().hex[:12]
        filepath = os.path.join(self.tmp.name, f"{did}{ext}")
        with open(filepath, "wb") as f:
            f.write(data)

        try:
            if ext == ".pdf" or data[:5] == b"%PDF-":
                pdf_doc = pypdfium2.PdfDocument(filepath)
                try:
                    num_pages = len(pdf_doc)
                finally:
                    pdf_doc.close()
            else:
                with Image.open(filepath) as im:
                    num_pages = max(1, int(getattr(im, "n_frames", 1)))
        except Exception as exc:
            try:
                os.remove(filepath)
            except OSError:
                pass
            raise ValueError(f"Could not read document: {exc}") from exc

        with self.docs_lock:
            self.docs[did] = {
                "id": did,
                "path": filepath,
                "filename": raw_name,
                "ext": ext,
                "pages": max(1, num_pages),
            }
            while len(self.docs) > 24:
                old_id = next(iter(self.docs))
                old_info = self.docs.pop(old_id, None)
                if old_info and os.path.exists(old_info["path"]):
                    try:
                        os.remove(old_info["path"])
                    except OSError:
                        pass

        return {
            "id": did,
            "pages": max(1, num_pages),
            "filename": raw_name,
        }

    def _render_page_images(
        self, filepath: str, ext: str, page_idx: int, highres_dpi: int
    ) -> tuple[Image.Image, Image.Image]:
        lowres_dpi = min(settings.IMAGE_DPI, highres_dpi)
        if ext == ".pdf":
            pdf_doc = pypdfium2.PdfDocument(filepath)
            try:
                page = pdf_doc[page_idx]
                try:
                    bmp_hi = page.render(scale=highres_dpi / 72.0)
                    try:
                        hi_img = bmp_hi.to_pil().convert("RGB")
                    finally:
                        bmp_hi.close()
                    bmp_lo = page.render(scale=lowres_dpi / 72.0)
                    try:
                        lo_img = bmp_lo.to_pil().convert("RGB")
                    finally:
                        bmp_lo.close()
                finally:
                    page.close()
            finally:
                pdf_doc.close()
            return lo_img, hi_img

        with Image.open(filepath) as im:
            if getattr(im, "n_frames", 1) > 1 and page_idx < im.n_frames:
                im.seek(page_idx)
            hi_img = im.convert("RGB")
        lo_img = hi_img.copy()
        return lo_img, hi_img

    def page(
        self, did: str, n: int, opts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        opts = opts or {}
        with self.docs_lock:
            doc_info = self.docs.get(did)
        if doc_info is None:
            raise ValueError(
                "Document not found — please drop or select the file again."
            )
        filepath = doc_info["path"]
        ext = doc_info["ext"]
        total_pages = doc_info["pages"]
        if n < 0 or n >= total_pages:
            raise ValueError(f"Page index {n} out of range (0..{total_pages - 1}).")

        mode = str(opts.get("mode") or "full_page").lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        skip_table_detection = bool(opts.get("skip_table_detection", False))
        include_figures = bool(opts.get("include_figures", False))
        skip_labels = [] if include_figures else None
        dpi = int(
            max(72, min(300, float(opts.get("dpi") or settings.IMAGE_DPI_HIGHRES)))
        )

        t_start = time.perf_counter()
        lo_img, hi_img = self._render_page_images(filepath, ext, n, dpi)
        orig_w, orig_h = hi_img.size

        scale = min(1.0, MAX_SIDE / max(orig_w, orig_h, 1))
        if scale < 1.0:
            disp_w = max(1, int(round(orig_w * scale)))
            disp_h = max(1, int(round(orig_h * scale)))
            display_img = hi_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        else:
            disp_w, disp_h = orig_w, orig_h
            display_img = hi_img

        jpg_buf = io.BytesIO()
        display_img.save(jpg_buf, format="JPEG", quality=84)
        img_data_url = "data:image/jpeg;base64," + base64.b64encode(
            jpg_buf.getvalue()
        ).decode("ascii")

        preds = self.get_predictors()
        layout_ms = 0.0
        ocr_ms = 0.0
        blocks_out: List[Dict[str, Any]] = []
        html_parts: List[str] = []
        raw_parts: List[str] = []
        page_json: Dict[str, Any] = {}

        if mode == "layout":
            t_l0 = time.perf_counter()
            layout_res = preds["layout"](
                [lo_img], target_image_sizes=[hi_img.size]
            )[0]
            layout_ms = (time.perf_counter() - t_l0) * 1000.0
            page_json = layout_res.model_dump(mode="json")
            for idx_b, b in enumerate(layout_res.bboxes):
                scaled_box = _scale_bbox_rect(
                    b.bbox, (orig_w, orig_h), (disp_w, disp_h)
                )
                blocks_out.append(
                    {
                        "id": str(idx_b),
                        "label": b.label,
                        "raw_label": b.raw_label,
                        "box": scaled_box,
                        "text": f"{b.label} (token budget ~{b.count})",
                        "html": "",
                        "confidence": round(float(b.confidence or 0.0), 3),
                        "count": b.count,
                        "skipped": False,
                        "error": False,
                    }
                )
                html_parts.append(
                    f'<div data-label="{b.label}" data-bbox="{int(b.bbox[0])} {int(b.bbox[1])} {int(b.bbox[2])} {int(b.bbox[3])}">'
                    f"<b>#{b.position + 1} {b.label}</b> — token estimate: {b.count}</div>"
                )

        elif mode == "table_rec":
            if skip_table_detection:
                table_imgs = [hi_img]
                table_bboxes = [[0.0, 0.0, float(orig_w), float(orig_h)]]
                table_counts = [0]
            else:
                t_l0 = time.perf_counter()
                layout_res = preds["layout"](
                    [lo_img], target_image_sizes=[hi_img.size]
                )[0]
                layout_ms = (time.perf_counter() - t_l0) * 1000.0
                tables = [
                    b
                    for b in layout_res.bboxes
                    if b.label in ("Table", "TableOfContents")
                ]
                table_bboxes = [list(expand_bbox(b.bbox)) for b in tables]
                table_imgs = [hi_img.crop(tuple(b)) for b in table_bboxes]
                table_counts = [b.count for b in tables]

            t_o0 = time.perf_counter()
            table_results = (
                preds["table_rec"](table_imgs, counts=table_counts)
                if table_imgs
                else []
            )
            ocr_ms = (time.perf_counter() - t_o0) * 1000.0
            page_json = {
                "tables": [tr.model_dump(mode="json") for tr in table_results]
            }
            for idx_t, (tbbox, tr) in enumerate(zip(table_bboxes, table_results)):
                scaled_box = _scale_bbox_rect(
                    tbbox, (orig_w, orig_h), (disp_w, disp_h)
                )
                clean_h = sanitize_ocr_html(tr.html or "")
                blocks_out.append(
                    {
                        "id": str(idx_t),
                        "label": "Table",
                        "raw_label": "Table",
                        "box": scaled_box,
                        "text": _strip_html_to_text(clean_h) or "(empty table)",
                        "html": clean_h,
                        "confidence": 1.0,
                        "count": 0,
                        "skipped": False,
                        "error": bool(tr.error),
                    }
                )
                if clean_h:
                    html_parts.append(clean_h)
                    raw_parts.append(clean_h)

        else:
            if mode == "block_ocr":
                t_l0 = time.perf_counter()
                layout_res = preds["layout"](
                    [lo_img], target_image_sizes=[hi_img.size]
                )[0]
                layout_ms = (time.perf_counter() - t_l0) * 1000.0
                t_o0 = time.perf_counter()
                page_res = preds["recognition"](
                    [hi_img],
                    [layout_res],
                    full_page=False,
                    skip_labels=skip_labels,
                )[0]
                ocr_ms = (time.perf_counter() - t_o0) * 1000.0
            else:
                t_o0 = time.perf_counter()
                page_res = preds["recognition"](
                    [hi_img], full_page=True, skip_labels=skip_labels
                )[0]
                ocr_ms = (time.perf_counter() - t_o0) * 1000.0

            page_json = page_res.model_dump(mode="json")
            for idx_b, blk in enumerate(page_res.blocks):
                scaled_box = _scale_bbox_rect(
                    blk.bbox, (orig_w, orig_h), (disp_w, disp_h)
                )
                clean_h = sanitize_ocr_html(blk.html or "")
                blocks_out.append(
                    {
                        "id": str(idx_b),
                        "label": blk.label,
                        "raw_label": blk.raw_label,
                        "box": scaled_box,
                        "text": _strip_html_to_text(clean_h)
                        or ("(skipped visual block)" if blk.skipped else ""),
                        "html": clean_h,
                        "confidence": round(float(blk.confidence or 0.0), 3),
                        "count": 0,
                        "skipped": bool(blk.skipped),
                        "error": bool(blk.error),
                    }
                )
                if not blk.skipped and clean_h:
                    x0, y0, x1, y1 = (int(c) for c in blk.bbox)
                    html_parts.append(
                        f'<div class="surya-blk" data-label="{blk.label}" data-bbox="{x0} {y0} {x1} {y1}">{clean_h}</div>'
                    )
                    raw_parts.append(
                        f'<div data-bbox="{x0} {y0} {x1} {y1}" data-label="{blk.label}">{clean_h}</div>'
                    )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        timings = {
            "layout_ms": round(layout_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "total_ms": round(total_ms, 1),
        }
        preview_html = sanitize_ocr_html("\n".join(html_parts))
        raw_html = "\n".join(raw_parts) if raw_parts else preview_html

        return {
            "page": n + 1,
            "page_index": n,
            "w": disp_w,
            "h": disp_h,
            "img": img_data_url,
            "mode_used": mode,
            "blocks": blocks_out,
            "preview_html": preview_html,
            "html": raw_html,
            "json": page_json,
            "timings": timings,
        }


def render_viewer_html() -> str:
    return PAGE.replace("__LOGO__", LOGO)


def make_handler(engine: ViewerEngine):
    """Standard library HTTP handler for running the Surya viewer."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(
            self, code: int, body: bytes, ctype: str = "application/json"
        ):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(
                    200,
                    render_viewer_html().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/info":
                self._send(200, json.dumps(engine.info()).encode("utf-8"))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                if self.path == "/upload":
                    fname = urllib.parse.unquote(
                        self.headers.get("X-Filename", "document.pdf")
                    )
                    res = engine.upload(body, filename=fname)
                elif self.path == "/page":
                    q = json.loads(body)
                    res = engine.page(
                        q["id"], int(q["n"]), q.get("opts", {})
                    )
                else:
                    return self._send(404, b"not found", "text/plain")
                self._send(200, json.dumps(res).encode("utf-8"))
            except Exception as exc:
                logger.warning(f"Viewer request error: {exc}")
                self._send(
                    400, json.dumps({"error": str(exc)}).encode("utf-8")
                )

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surya 2.5 Viewer · Datalab</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
:root{
 --font-body:"Geist",ui-sans-serif,system-ui,sans-serif;--font-mono:"Geist Mono",ui-monospace,"SF Mono",Menlo,monospace;
 --page:oklch(98.5% .004 263);--surface:oklch(96.5% .006 263);--surface-2:oklch(94% .007 263);
 --rule:oklch(86% .008 263);--rule-strong:oklch(78% .01 263);
 --ink:oklch(18% .012 263);--ink-2:oklch(36% .012 263);--ink-3:oklch(48% .014 263);--ink-4:oklch(70% .014 263);
 --accent:oklch(38% .24 263);--accent-soft:oklch(96% .04 263);--accent-strong:oklch(30% .26 263);--accent-ink:oklch(99% 0 0);
 --ok:oklch(58% .15 145);--warn:oklch(72% .16 75);--err:oklch(56% .2 25);--focus:oklch(38% .24 263 / .35);
 --hairline:1px;--ease:cubic-bezier(.25,1,.5,1)}
@media (min-resolution:2dppx){:root{--hairline:.5px}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 var(--font-body);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.mono{font-family:var(--font-mono)}
.label{font:500 10.5px/1 var(--font-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
/* top bar */
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;height:56px;padding:0 24px;background:color-mix(in oklch,var(--page) 88%,transparent);backdrop-filter:blur(8px);border-bottom:var(--hairline) solid var(--rule)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em}
.mark{width:30px;height:30px;margin:0 -4px}
.brand .sep{color:var(--ink-4);font-weight:400}.brand .prod{font-weight:500}
.ver{font:11px var(--font-mono);color:var(--ink-3);border:var(--hairline) solid var(--rule);border-radius:999px;padding:2px 8px}
.top .sp{flex:1}.top nav{display:flex;align-items:center;gap:18px;font-size:13px}.top nav a{color:var(--ink-2)}
.cta{background:var(--accent);color:var(--accent-ink)!important;padding:7px 12px;border-radius:6px;font-weight:500}.cta:hover{background:var(--accent-strong);text-decoration:none}
/* layout */
.wrap{display:grid;grid-template-columns:316px minmax(0,1fr);min-height:calc(100vh - 56px)}
aside{border-right:var(--hairline) solid var(--rule);padding:20px;display:flex;flex-direction:column;gap:20px;position:sticky;top:56px;height:calc(100vh - 56px);overflow:auto;background:var(--page)}
main{padding:24px;display:flex;flex-direction:column;gap:24px;min-width:0}
@media(max-width:920px){.wrap{grid-template-columns:1fr}aside{position:static;height:auto;border-right:0;border-bottom:var(--hairline) solid var(--rule)}}
/* drop zone */
#drop{border:1px dashed var(--rule-strong);border-radius:10px;padding:18px 14px;text-align:center;cursor:pointer;background:var(--surface);transition:border-color var(--ease) .12s,background .12s}
#drop:hover,#drop.over{border-color:var(--accent);background:var(--accent-soft)}
#drop b{display:block;font-weight:500;margin-bottom:2px}#drop span{color:var(--ink-3);font-size:12px}
#fname{margin-top:8px;font:12px var(--font-mono);color:var(--ink-2);word-break:break-all}
/* controls */
.group{display:flex;flex-direction:column;gap:10px}
.seg{display:grid;grid-template-columns:repeat(4,1fr);border:var(--hairline) solid var(--rule-strong);border-radius:7px;overflow:hidden}
.seg button{border:0;background:transparent;padding:7px 2px;font:500 11.5px var(--font-body);color:var(--ink-2);cursor:pointer;white-space:nowrap}
.seg button+button{border-left:var(--hairline) solid var(--rule)}.seg button.on{background:var(--ink);color:var(--page)}
.hint{font-size:12px;color:var(--ink-3);line-height:1.4}
.sw{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:13px;color:var(--ink-2);cursor:pointer;user-select:none}
.sw input{appearance:none;width:30px;height:18px;border-radius:999px;background:var(--rule-strong);position:relative;cursor:pointer;transition:background .15s;flex:none}
.sw input::after{content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#fff;transition:transform .15s var(--ease)}
.sw input:checked{background:var(--accent)}.sw input:checked::after{transform:translateX(12px)}
.sw input:focus-visible{outline:3px solid var(--focus)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.field{display:flex;flex-direction:column;gap:5px}.field select,.field input{font:13px var(--font-body);padding:6px 8px;border:var(--hairline) solid var(--rule-strong);border-radius:6px;background:#fff;color:var(--ink);width:100%}
.btn{border:0;background:var(--accent);color:var(--accent-ink);font:500 13px var(--font-body);padding:9px 12px;border-radius:7px;cursor:pointer;text-align:center}
.btn:hover{background:var(--accent-strong)}.btn:disabled{background:var(--rule-strong);cursor:default}
.btn-sec{border:var(--hairline) solid var(--rule-strong);background:#fff;color:var(--ink-2);font:500 12px var(--font-body);padding:6px 10px;border-radius:6px;cursor:pointer}
.btn-sec:hover{background:var(--surface)}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:0}
#stats{font:12px/1.7 var(--font-mono);color:var(--ink-2)}#stats b{color:var(--ink);font-weight:500}
#status{font:500 12px var(--font-mono);color:var(--ink-2);min-height:18px;font-variant-numeric:tabular-nums}
/* progress banner */
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.pbanner{position:sticky;top:68px;z-index:4;background:#fff;border:1.5px solid var(--accent);border-radius:12px;padding:14px 18px;box-shadow:0 6px 20px oklch(18% .012 263 / .10);display:flex;flex-direction:column;gap:10px;transition:border-color .2s,background .2s}
.pbanner[hidden]{display:none!important}
.pbanner.done{border-color:var(--ok);background:oklch(98.5% .015 145)}
.pbanner.err{border-color:var(--err);background:oklch(98% .02 25)}
.prow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.spin{width:18px;height:18px;border:2.5px solid var(--accent-soft);border-top-color:var(--accent);border-radius:50%;animation:spin .85s linear infinite;flex:none;will-change:transform}
.pbanner.done .spin,.pbanner.err .spin{display:none}
.pbadge{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;font:700 12px var(--font-mono);color:#fff;flex:none}
.pbanner.done .pbadge{background:var(--ok)}
.pbanner.err .pbadge{background:var(--err)}
.ptitle{font:600 14px var(--font-body);color:var(--ink)}
.pbanner.err .ptitle{color:var(--err)}
.psub{font:500 12.5px var(--font-mono);color:var(--ink-2);margin-left:auto;font-variant-numeric:tabular-nums}
.ptrack{height:8px;border-radius:999px;background:var(--surface-2);overflow:hidden;position:relative}
.pfill{height:100%;border-radius:999px;background:var(--accent);transition:width .25s var(--ease)}
.pfill.indet{width:100%!important;background:linear-gradient(90deg,var(--accent-soft) 0%,var(--accent) 50%,var(--accent-soft) 100%);background-size:200% 100%;animation:shimmer 1.4s ease-in-out infinite}
.pbanner.done .pfill{background:var(--ok)}
.pbanner.err .pfill{background:var(--err)}
.inflight{border:1.5px dashed var(--accent);border-radius:12px;padding:36px 24px;text-align:center;background:color-mix(in oklch,var(--accent-soft) 45%,#fff);color:var(--ink-2);display:flex;flex-direction:column;align-items:center;gap:10px}
.inflight b{font-size:15px;color:var(--ink)}
.callout{border:var(--hairline) solid var(--rule);border-radius:10px;padding:12px;background:var(--surface);font-size:12.5px;color:var(--ink-2)}
/* empty state */
.empty{border:1px dashed var(--rule-strong);border-radius:14px;padding:64px 24px;text-align:center;color:var(--ink-3)}
.empty h2{font-weight:500;color:var(--ink);letter-spacing:-.015em;margin:0 0 6px;font-size:22px}
/* page card */
.card{background:#fff;border:var(--hairline) solid var(--rule);border-radius:12px;overflow:hidden}
.ch{display:flex;align-items:center;flex-wrap:wrap;gap:10px 14px;padding:12px 16px;border-bottom:var(--hairline) solid var(--rule)}
.ch h3{margin:0;font-size:15px;font-weight:600;letter-spacing:-.01em}
.chip{font:500 10.5px/1 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;padding:5px 7px;border-radius:5px;background:var(--accent-soft);color:var(--accent)}
.meta{font:12px var(--font-mono);color:var(--ink-3)}.ch .sp{flex:1}
.tbar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:var(--surface-2);width:170px}.tbar i{display:block;height:100%}
.body{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);min-height:440px}
@media(max-width:1200px){.body{grid-template-columns:1fr}}
.pageview{padding:16px;background:var(--surface);border-right:var(--hairline) solid var(--rule)}
svg.pg{width:100%;height:auto;display:block;background:#fff;box-shadow:0 1px 2px oklch(18% .012 263 / .08),0 0 0 var(--hairline) var(--rule)}
.right{display:flex;flex-direction:column;min-width:0}
.tabs{display:flex;align-items:center;gap:2px;padding:8px 12px 0;border-bottom:var(--hairline) solid var(--rule);overflow-x:auto}
.tabs button{border:0;background:transparent;padding:8px 10px;font:500 12.5px var(--font-body);color:var(--ink-3);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
.tabs .sp{flex:1}
.tabs .copy-btn{font:500 11px var(--font-mono);padding:4px 8px;margin-bottom:4px;border:var(--hairline) solid var(--rule);border-radius:5px;background:var(--surface);color:var(--ink-2)}
.tabs .copy-btn:hover{background:var(--surface-2)}
.pane{flex:1;overflow:auto;max-height:78vh;padding:14px 16px}
.pane.src{white-space:pre-wrap;font:12.5px/1.6 var(--font-mono);color:var(--ink-2)}
.preview{font-size:14px;line-height:1.6}
.preview h1,.preview h2,.preview h3,.preview h4{letter-spacing:-.015em;line-height:1.25;margin:1em 0 .4em}
.preview h1{font-size:20px}.preview h2{font-size:17px}.preview h3{font-size:15px}
.preview p{margin:.5em 0}.preview ul,.preview ol{padding-left:1.4em}
.preview table{border-collapse:collapse;margin:10px 0;font-size:12.5px;width:auto;max-width:100%}
.preview td,.preview th{border:var(--hairline) solid var(--rule-strong);padding:4px 7px;vertical-align:top}.preview th{background:var(--surface)}
.preview math{font:12.5px var(--font-mono);background:var(--accent-soft);color:var(--accent-strong);padding:1px 5px;border-radius:4px}
.preview math[display="block"]{display:block;padding:8px 10px;margin:8px 0;overflow-x:auto}
.bk-row{display:flex;align-items:baseline;gap:10px;padding:4px 6px;border-radius:5px;font-size:13px;cursor:default}
.bk-row:hover,.bk-row.on{background:var(--accent-soft)}
.bk-row .k{font:11px var(--font-mono);color:var(--ink-4);min-width:26px;text-align:right}
.bk-row .t{flex:1;min-width:0;word-break:break-word}
.bk-row .btag{font:500 10.5px var(--font-mono);padding:2px 6px;border-radius:4px;background:var(--surface-2);color:var(--ink-2);white-space:nowrap}
.bk-row.ignored{opacity:.55}
/* overlays */
rect.bk{fill:transparent;stroke-width:2;vector-effect:non-scaling-stroke;cursor:pointer}
rect.bk.on{fill:oklch(55% .2 263 / .12);stroke-width:3.5}
text.bl{font-family:var(--font-mono);font-weight:600;paint-order:stroke;stroke:#fff;pointer-events:none}
path.ord{fill:none;stroke:oklch(60% .2 30);stroke-width:1.5;vector-effect:non-scaling-stroke;opacity:.82;pointer-events:none}
circle.on-n{fill:oklch(60% .2 30);pointer-events:none}
text.n{font-family:var(--font-mono);font-weight:600;fill:#fff;text-anchor:middle;dominant-baseline:central;pointer-events:none}
.hide-blocks rect.bk,.hide-blocks text.bl,.hide-labels text.bl,.hide-order .ord,.hide-order circle.on-n,.hide-order text.n{display:none}
footer{padding:18px 24px;border-top:var(--hairline) solid var(--rule);display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--ink-3)}
</style></head><body>
<div class="top">
 <div class="brand"><img class="mark" src="data:image/png;base64,__LOGO__" alt="Datalab">Datalab<span class="sep">/</span><span class="prod">Surya 2.5</span></div>
 <span class="ver" id="ver"></span><span class="ver" id="mdl" hidden></span>
 <div class="sp"></div>
 <nav><a href="https://github.com/datalab-to/surya" target="_blank" rel="noopener">GitHub</a><a href="https://documentation.datalab.to" target="_blank" rel="noopener">Docs</a><a class="cta" href="https://www.datalab.to/?utm_source=surya-viewer" target="_blank" rel="noopener">Try Chandra on Datalab</a></nav>
</div>
<div class="wrap">
<aside>
 <div>
  <div id="drop"><b>Drop a PDF or image</b><span>PDF, PNG, JPG, WEBP, GIF, TIFF, BMP</span><input id="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.tiff,.tif,.bmp" hidden></div>
  <div id="fname"></div>
 </div>
 <div class="group"><div class="label">Mode</div>
  <div class="seg" id="mode">
   <button data-v="full_page">Full OCR</button>
   <button data-v="block_ocr">Block OCR</button>
   <button data-v="layout">Layout</button>
   <button data-v="table_rec">Tables</button>
  </div>
  <div class="hint" id="modehint"></div>
 </div>
 <div class="group"><div class="label">Options</div>
  <label class="sw">Include Figure/Picture OCR<input type="checkbox" id="include_figures"></label>
  <label class="sw">Skip table detection<input type="checkbox" id="skip_tbl_det"></label>
  <div class="row2">
   <div class="field"><span class="label">DPI</span><select id="dpi"><option>96</option><option>150</option><option selected>192</option><option>300</option></select></div>
   <div class="field"><span class="label">Max pages</span><input id="maxp" type="number" min="1" value="20"></div>
  </div>
  <div class="field"><span class="label">Page range (0-indexed)</span><input id="prange" type="text" placeholder="All (or e.g. 0,2-5)"></div>
  <button class="btn" id="rerun" disabled>Run again with these options</button>
 </div>
 <div class="group"><div class="label">Show</div>
  <label class="sw">Layout blocks<input type="checkbox" class="disp" data-c="blocks" checked></label>
  <label class="sw">Block labels<input type="checkbox" class="disp" data-c="labels" checked></label>
  <label class="sw">Reading order<input type="checkbox" class="disp" data-c="order" checked></label>
  <div id="legend" class="hint"></div>
 </div>
 <div class="group"><div class="label">Run</div>
  <div id="status"></div>
  <div id="stats"></div>
  <button class="btn-sec" id="dl-json" hidden>Download full JSON (.json)</button>
 </div>
 <div class="callout">Need handwriting, complex forms, and highest-accuracy VLM parsing at scale? <a href="https://www.datalab.to/?utm_source=surya-viewer" target="_blank" rel="noopener">Chandra on Datalab</a> includes 5,000 free pages/month with a work email.</div>
</aside>
<main>
 <div class="pbanner" id="pbanner" hidden>
  <div class="prow">
   <div class="spin" id="pspin"></div>
   <div class="pbadge" id="pbadge" hidden>✓</div>
   <div class="ptitle" id="ptitle">Running Surya 2.5…</div>
   <div class="psub" id="psub">0 / 0 pages · 0.0s</div>
   <button class="btn-sec" id="pbanner-dl" hidden>Download .json</button>
  </div>
  <div class="ptrack"><div class="pfill indet" id="pfill" style="width:0%"></div></div>
 </div>
 <div id="out" style="display:flex;flex-direction:column;gap:24px"><div class="empty"><h2>Inspect a document with Surya 2.5</h2>Drop a PDF or image on the left. Pages appear as they finish with interactive layout boxes, reading order, typeset KaTeX math, HTML tables, and JSON.</div></div>
</main>
</div>
<footer><span>Surya 2.5 · multilingual document OCR &amp; layout VLM by <a href="https://www.datalab.to" target="_blank" rel="noopener">Datalab</a></span><span class="mono">Runs locally: your files stay on this machine.</span></footer>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
const BLOCK={
 "Text":"oklch(55% .02 263)",
 "SectionHeader":"oklch(45% .22 263)",
 "ListGroup":"oklch(58% .15 145)",
 "Table":"oklch(55% .18 320)",
 "Form":"oklch(50% .16 25)",
 "TableOfContents":"oklch(55% .12 180)",
 "Equation":"oklch(52% .22 300)",
 "Picture":"oklch(60% .15 160)",
 "Figure":"oklch(60% .15 160)",
 "Diagram":"oklch(60% .15 160)",
 "Caption":"oklch(62% .14 200)",
 "Footnote":"oklch(60% .12 60)",
 "Code":"oklch(48% .18 230)",
 "PageHeader":"oklch(68% .03 263)",
 "PageFooter":"oklch(68% .03 263)"
};
const LEGEND_ITEMS=[
 ["Text","oklch(55% .02 263)"],
 ["SectionHeader","oklch(45% .22 263)"],
 ["ListGroup","oklch(58% .15 145)"],
 ["Table","oklch(55% .18 320)"],
 ["Equation","oklch(52% .22 300)"],
 ["Picture/Figure","oklch(60% .15 160)"],
 ["Caption/Footnote","oklch(62% .14 200)"],
 ["Header/Footer","oklch(68% .03 263)"]
];
const STAGES=[
 ["layout_ms","Layout","oklch(45% .22 263)"],
 ["ocr_ms","VLM OCR","oklch(58% .15 145)"]
];
const HINT={
 full_page:"Single VLM pass per page returning layout bboxes, reading order, and HTML (with automatic block fallback).",
 block_ocr:"Two-stage pipeline: 96 DPI layout detection followed by per-block crop OCR.",
 layout:"Layout detection only: bounding boxes, canonical labels, reading order, and token-count estimates.",
 table_rec:"HTML table recognition (with colspan/rowspan). Toggle 'Skip table detection' if the image is already a cropped table."
};
let mode="full_page",doc=null,stats=[],pagesJson=[],runId=0,timerId=null,runAbort=null;
function setMode(v){mode=v;$$('#mode button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));$('#modehint').textContent=HINT[v]||''}
$$('#mode button').forEach(b=>b.onclick=()=>{setMode(b.dataset.v);if(doc)run();});
$('#legend').innerHTML=LEGEND_ITEMS.map(([k,c])=>`<span style="white-space:nowrap;margin-right:8px"><span class="dot" style="background:${c}"></span>${k}</span>`).join(' ');
function applyDisplay(){const o=$('#out');$$('.disp').forEach(i=>o.classList.toggle('hide-'+i.dataset.c,!i.checked))}
$$('.disp').forEach(i=>i.onchange=applyDisplay);applyDisplay();

const drop=$('#drop'),file=$('#file');
drop.onclick=()=>file.click();file.onchange=()=>file.files[0]&&start(file.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');e.dataTransfer.files[0]&&start(e.dataTransfer.files[0])};
$('#rerun').onclick=()=>doc&&run();
function dlJson(){
 if(!pagesJson.length)return;
 const blob=new Blob([JSON.stringify(pagesJson,null,2)],{type:'application/json;charset=utf-8'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download=((doc&&doc.name)||'document').replace(/\.[^.]+$/,'')+'_surya.json';a.click();
}
$('#dl-json').onclick=dlJson;
$('#pbanner-dl').onclick=dlJson;

function setBanner(state,title,sub,pct,indet){
 const b=$('#pbanner'),f=$('#pfill'),bd=$('#pbadge');
 b.hidden=false;
 b.classList.toggle('done',state==='done');
 b.classList.toggle('err',state==='err');
 bd.hidden=state==='run';
 bd.textContent=state==='err'?'!':'✓';
 $('#ptitle').textContent=title;
 $('#psub').textContent=sub||'';
 f.classList.toggle('indet',!!indet);
 f.style.width=Math.max(0,Math.min(100,pct||0))+'%';
 $('#pbanner-dl').hidden=!(state==='done'&&pagesJson.length);
 $('#status').textContent=title+(sub?' ('+sub+')':'');
}

function stopTimer(){if(timerId){clearInterval(timerId);timerId=null;}}

fetch('/info').then(r=>r.json()).then(i=>{
 $('#ver').textContent='v'+(i.version||'0.24.0');
 if(i.model){$('#mdl').textContent=String(i.model);$('#mdl').hidden=false;}
 const d=i.defaults||{};
 setMode(d.mode||'full_page');
 $('#include_figures').checked=!!d.include_figures;
 $('#skip_tbl_det').checked=!!d.skip_table_detection;
 if(d.dpi)$('#dpi').value=String(Math.round(d.dpi));
 if(d.max_pages)$('#maxp').value=d.max_pages;
}).catch(()=>setMode('full_page'));
setMode('full_page');

function parsePageIndices(spec,total,maxP){
 spec=(spec||'').trim();
 if(!spec){const N=Math.min(total,Math.max(1,maxP));return Array.from({length:N},(_,i)=>i);}
 const out=[],seen=new Set();
 for(const part of spec.split(',')){
  const s=part.trim();if(!s)continue;
  if(s.includes('-')){
   const [aStr,bStr]=s.split('-',2);
   const a=Math.max(0,parseInt(aStr,10)||0),b=Math.min(total-1,parseInt(bStr,10)||0);
   for(let p=a;p<=b;p++){if(p>=0&&p<total&&!seen.has(p)){seen.add(p);out.push(p);}}
  }else{
   const p=parseInt(s,10);
   if(!Number.isNaN(p)&&p>=0&&p<total&&!seen.has(p)){seen.add(p);out.push(p);}
  }
 }
 return out.slice(0,Math.max(1,maxP));
}

function abortRun(){if(runAbort){try{runAbort.abort();}catch(_){}runAbort=null;}}

async function start(f){
 stopTimer();abortRun();
 $('#fname').textContent=f.name+' · '+(f.size/1024).toFixed(0)+' KB';
 setBanner('run',`Uploading ${f.name}…`,(f.size/1024).toFixed(0)+' KB',100,true);
 try{
  const r=await fetch('/upload',{method:'POST',headers:{'X-Filename':encodeURIComponent(f.name)},body:await f.arrayBuffer()});
  const u=await r.json();
  if(!r.ok||u.error||u.detail){setBanner('err','Upload error: '+(u.error||u.detail||r.statusText),'',100,false);return}
  doc={...u,name:f.name};$('#rerun').disabled=false;run();
 }catch(err){
  setBanner('err','Upload failed: '+(err&&err.message?err.message:String(err)),'',100,false);
 }
}

function opts(){
 return{
  mode,
  include_figures:$('#include_figures').checked,
  skip_table_detection:$('#skip_tbl_det').checked,
  dpi:+$('#dpi').value
 };
}

function typesetMath(container){
 if(!window.katex)return;
 container.querySelectorAll('math').forEach(el=>{
  const tex=el.textContent||'';
  const disp=el.getAttribute('display')==='block';
  try{
   const span=document.createElement(disp?'div':'span');
   window.katex.render(tex,span,{displayMode:disp,throwOnError:false});
   el.replaceWith(span);
  }catch(_){}
 });
}

async function run(){
 stopTimer();abortRun();
 const id=++runId,ctrl=new AbortController(),o=opts(),maxP=Math.max(1,+$('#maxp').value||20);
 runAbort=ctrl;
 const pageList=parsePageIndices($('#prange').value,doc.pages,maxP);
 if(!pageList.length){setBanner('err','No valid pages in range (0..'+(doc.pages-1)+')','',100,false);return;}
 $('#out').innerHTML='';stats=[];pagesJson=[];$('#dl-json').hidden=true;summary();
 const inflight=document.createElement('div');
 inflight.className='inflight';
 const infSpin=document.createElement('div');infSpin.className='spin';
 const infTitle=document.createElement('b');
 const infSub=document.createElement('span');infSub.className='mono';infSub.style.fontSize='12px';infSub.style.fontVariantNumeric='tabular-nums';
 inflight.append(infSpin,infTitle,infSub);
 const t0=performance.now();
 let curIdx=0,curPage=pageList[0]+1;
 const updateClock=()=>{
  const elapsed=((performance.now()-t0)/1000).toFixed(1)+'s';
  const pct=Math.round((curIdx/pageList.length)*100);
  const sub=`${curIdx} / ${pageList.length} pages (${pct}%) · ${elapsed}`;
  $('#psub').textContent=sub;
  infSub.textContent=`${elapsed} elapsed · mode: ${o.mode}`;
 };
 const updateStage=()=>{
  const pct=Math.round((curIdx/pageList.length)*100);
  const elapsed=((performance.now()-t0)/1000).toFixed(1)+'s';
  const sub=`${curIdx} / ${pageList.length} pages (${pct}%) · ${elapsed}`;
  setBanner('run',`Processing Page ${curPage} (${curIdx+1} of ${pageList.length}) · mode: ${o.mode}`,sub,pct,curIdx===0);
  infTitle.textContent=`Processing Page ${curPage} (${curIdx+1} of ${pageList.length})…`;
  infSub.textContent=`${elapsed} elapsed · mode: ${o.mode}`;
 };
 $('#out').appendChild(inflight);
 updateStage();
 timerId=setInterval(updateClock,250);
 for(let idx=0;idx<pageList.length;idx++){
  if(id!==runId)return;
  curIdx=idx;curPage=pageList[idx]+1;
  updateStage();
  const n=pageList[idx];
  let p;
  try{
   const resp=await fetch('/page',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:doc.id,n,opts:o}),signal:ctrl.signal});
   p=await resp.json();
  }catch(err){
   if(id!==runId||(err&&err.name==='AbortError'))return;
   stopTimer();inflight.remove();
   setBanner('err',`Error on page ${n+1}: ${err&&err.message?err.message:String(err)}`,`${idx} / ${pageList.length} pages completed`,100,false);
   return;
  }
  if(id!==runId)return;
  if(p.error||p.detail){
   stopTimer();
   inflight.remove();
   setBanner('err',`Error on page ${n+1}: ${p.error||p.detail}`,`${idx} / ${pageList.length} pages completed`,100,false);
   return;
  }
  render(p,inflight);
  stats.push(p.timings);pagesJson.push(p.json);$('#dl-json').hidden=!pagesJson.length;summary();
 }
 stopTimer();
 inflight.remove();
 const totMs=performance.now()-t0;
 const avgMs=(totMs/pageList.length).toFixed(0);
 const pps=(1000*pageList.length/Math.max(1,totMs)).toFixed(2);
 setBanner('done',`Processed ${pageList.length} page${pageList.length>1?'s':''} in ${(totMs/1000).toFixed(2)}s`,`${avgMs} ms/page · ${pps} pages/s`,100,false);
}

function summary(){
 const el=$('#stats');if(!stats.length){el.innerHTML='';return}
 const n=stats.length,s=k=>stats.reduce((a,t)=>a+(t[k]||0),0),tot=Math.max(1,s('total_ms'));
 el.innerHTML=`<b>${(tot/n).toFixed(0)} ms</b> per page · <b>${(1000*n/tot).toFixed(2)}</b> pages/s<br>`+
  STAGES.filter(([k])=>s(k)>0).map(([k,l])=>`${l} ${(s(k)/n).toFixed(0)} ms`).join(' · ');
}

function render(p,beforeEl){
 const c=document.createElement('div');c.className='card';const t=p.timings||{},tot=t.total_ms||1;
 const u=Math.max(1,p.w/640);
 const modeMap={full_page:'Full-Page OCR',block_ocr:'Block OCR',layout:'Layout',table_rec:'Table Rec'};
 const src=modeMap[p.mode_used]||p.mode_used;
 const bar=STAGES.map(([k,l,col])=>`<i style="width:${100*(t[k]||0)/tot}%;background:${col}" title="${l}: ${t[k]||0} ms"></i>`).join('');
 const blocks=p.blocks.map((b,k)=>{
  const col=BLOCK[b.label]||'oklch(55% .02 263)',[x0,y0,x1,y1]=b.box;
  return `<rect class="bk" data-b="${k}" x="${x0-2}" y="${y0-2}" width="${Math.max(2,x1-x0+4)}" height="${Math.max(2,y1-y0+4)}" stroke="${col}"/><text class="bl" data-b="${k}" x="${x0}" y="${Math.max(11*u,y0-4*u)}" fill="${col}" font-size="${10.5*u}" stroke-width="${3*u}">${esc(b.label)}</text>`;
 }).join('');
 const cen=p.blocks.map(it=>[(it.box[0]+it.box[2])/2,(it.box[1]+it.box[3])/2]);
 const ord=cen.length>1?`<path class="ord" d="M${cen.map(pt=>pt.map(v=>v.toFixed(1)).join(',')).join(' L')}"/>`:'';
 const nums=cen.map((pt,k)=>`<circle class="on-n" cx="${pt[0]}" cy="${pt[1]}" r="${7.5*u}"/><text class="n" x="${pt[0]}" y="${pt[1]}" font-size="${8.5*u}">${k+1}</text>`).join('');
 const blockRows=p.blocks.map((b,k)=>{
  const col=BLOCK[b.label]||'oklch(55% .02 263)';
  const snippet=(b.text||'').slice(0,220)||'(visual block)';
  const confStr=b.confidence?`${Math.round(b.confidence*100)}%`:'';
  return `<div class="bk-row ${b.skipped?'ignored':''}" data-b="${k}"><span class="k">${k+1}</span><span class="btag" style="border-left:3px solid ${col}">${esc(b.label)}</span><span class="t">${esc(snippet)}</span><span class="meta">${esc(confStr)}</span></div>`;
 }).join('');
 const tabs=['Preview','Blocks','HTML','JSON'];
 c.innerHTML=`<div class="ch"><h3>Page ${p.page}</h3><span class="chip">${esc(src)}</span><span class="meta">${p.blocks.length} blocks</span><span class="sp"></span><div class="tbar">${bar}</div><span class="meta">${t.total_ms} ms</span></div>
 <div class="body">
  <div class="pageview"><svg class="pg" viewBox="0 0 ${p.w} ${p.h}"><image href="${p.img}" width="${p.w}" height="${p.h}"/>${blocks}${ord}${nums}</svg></div>
  <div class="right">
   <div class="tabs">${tabs.map((n,i)=>`<button data-tab="${n}" class="${i?'':'on'}">${n}</button>`).join('')}<span class="sp"></span><button class="copy-btn" data-copy="1">Copy HTML</button></div>
   <div class="pane preview" data-pane="Preview">${p.preview_html||'<span class="meta">(Empty page)</span>'}</div>
   <div class="pane" data-pane="Blocks" hidden>${blockRows||'<span class="meta">No blocks detected.</span>'}</div>
   <div class="pane src" data-pane="HTML" hidden>${esc(p.html||'')}</div>
   <div class="pane src" data-pane="JSON" hidden>${esc(JSON.stringify(p.json||{},null,2))}</div>
  </div>
 </div>`;
 typesetMath(c.querySelector('[data-pane="Preview"]'));
 c.querySelectorAll('.tabs button[data-tab]').forEach(b=>b.onclick=()=>{
  c.querySelectorAll('.tabs button[data-tab]').forEach(x=>x.classList.toggle('on',x===b));
  c.querySelectorAll('.pane').forEach(x=>x.hidden=x.dataset.pane!==b.dataset.tab);
 });
 const copyBtn=c.querySelector('.copy-btn');
 if(copyBtn)copyBtn.onclick=()=>{navigator.clipboard.writeText(p.html||'');copyBtn.textContent='Copied!';setTimeout(()=>copyBtn.textContent='Copy HTML',1200);};
 const hlBlock=b=>c.querySelectorAll('[data-b]').forEach(el=>el.classList.toggle('on',el.dataset.b==b));
 c.querySelectorAll('.bk-row').forEach(el=>{el.onmouseenter=()=>hlBlock(el.dataset.b);el.onmouseleave=()=>hlBlock(-1)});
 c.querySelectorAll('rect.bk').forEach(el=>{
  el.onmouseenter=()=>{hlBlock(el.dataset.b);const r=c.querySelector(`.bk-row[data-b="${el.dataset.b}"]`);r&&r.scrollIntoView({block:'nearest'})};
  el.onmouseleave=()=>hlBlock(-1);
 });
 if(beforeEl&&beforeEl.parentNode===$('#out'))$('#out').insertBefore(c,beforeEl);
 else $('#out').appendChild(c);
}
</script></body></html>"""


@click.command(help="Launch the Surya 2.5 interactive web viewer.")
@click.version_option(version="0.24.0", prog_name="surya-ocr")
@click.option("--port", type=int, default=8000, help="Port to run the viewer on.")
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind the viewer to.")
@click.option(
    "--mode",
    type=click.Choice(["full_page", "block_ocr", "layout", "table_rec"]),
    default="full_page",
    help="Initial Surya mode.",
)
@click.option("--dpi", type=int, default=192, help="Initial render resolution (DPI).")
@click.option(
    "--max_pages",
    "--max-pages",
    type=int,
    default=20,
    help="Initial page limit per document.",
)
def viewer_cli(
    port: int = 8000,
    host: str = "127.0.0.1",
    mode: str = "full_page",
    dpi: int = 192,
    max_pages: int = 20,
):
    engine = ViewerEngine(
        defaults={
            "mode": mode,
            "skip_table_detection": False,
            "include_figures": False,
            "dpi": dpi,
            "max_pages": max_pages,
        }
    )
    srv = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"Surya Viewer running on http://{host}:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    viewer_cli()
