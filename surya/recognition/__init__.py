"""RecognitionPredictor: per-block OCR via BLOCK_PROMPT.

Given page images and corresponding LayoutResult (or any list of LayoutBox),
crops each block, runs BLOCK_PROMPT, returns PageOCRResult per page.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.common.blank import is_blank_region
from surya.inference import SuryaInferenceManager, get_default_manager
from surya.inference.parsers import clean_block_html, parse_full_page_html
from surya.inference.prompts import (
    PROMPT_TYPE_BLOCK,
    PROMPT_TYPE_HIGH_ACCURACY_BBOX,
    SKIP_OCR_LABELS,
)
from surya.inference.schema import BatchInputItem
from surya.inference.util import image_token_budget
from surya.layout.label import LAYOUT_PRED_RELABEL, TEXT_LABELS
from surya.layout.schema import LayoutResult
from surya.logging import get_logger
from surya.recognition.schema import (
    BlockOCRResult,
    PageOCRResult,
)
from surya.settings import settings

logger = get_logger()


# Surya's canonical labels we shouldn't OCR (mirrors model-emitted SKIP_OCR_LABELS
# after canonicalization).
SKIP_CANON_LABELS = {LAYOUT_PRED_RELABEL.get(lbl, lbl) for lbl in SKIP_OCR_LABELS}

# Full-page OCR regeneration schedule (chandra-style), used ONLY when
# settings.SURYA_FULLPAGE_REGEN is True. Round 0 is greedy; a page whose output
# loops / fails to parse is re-requested at escalating temperature (top_p 0.95)
# before resorting to the (slower) block-mode fallback. Mirrors chandra's
# retry_temperature = min(0.2*(n+1), 0.8) over MAX_VLLM_RETRIES=6 retries.
_REGEN_ROUNDS = [(0.0, None)] + [(min(0.2 * (n + 1), 0.8), 0.95) for n in range(6)]


def _crop_block(image: Image.Image, polygon, pad: int = 4) -> Image.Image:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(image.size[0], int(max(xs)) + pad)
    y1 = min(image.size[1], int(max(ys)) + pad)
    if x1 <= x0 or y1 <= y0:
        return image.crop((0, 0, 1, 1))
    return image.crop((x0, y0, x1, y1))


def _drop_blank_text_blocks(
    image: Image.Image,
    blocks: List[BlockOCRResult],
) -> List[BlockOCRResult]:
    """Drop text-labeled blocks whose source page region is essentially blank.

    Full-page OCR can emit text divs for regions that are visually empty
    (margins, gutter space) — the model hallucinates a paragraph where there
    is none. We crop the region, count near-white pixels, and drop the block
    when the fraction exceeds ``blank_pixel_fraction``. Only text-like labels
    (see ``TEXT_LABELS``) are eligible: tables, forms, equations, and visual
    blocks may legitimately contain large whitespace and are left untouched.
    """
    kept: List[BlockOCRResult] = []
    dropped = 0
    for blk in blocks:
        if blk.label not in TEXT_LABELS or blk.skipped or blk.error:
            kept.append(blk)
            continue
        crop = _crop_block(image, blk.polygon)
        if not is_blank_region(crop):
            kept.append(blk)
            continue
        dropped += 1
    if dropped:
        logger.info(f"dropped {dropped} blank text block(s) from full-page OCR")
    return kept


def _detect_repeat_loop(
    text: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    scaling_factor: float = 3.0,
) -> bool:
    """True iff the tail of ``text`` ends in a repeating sequence.

    Ported from chandra's detect_repeat_token. For each candidate length
    1..window_size/2, takes that many trailing chars and counts consecutive
    identical preceding blocks. Shorter loops need many repeats to count;
    longer ones only need a few. Catches the typical decoder failure mode
    where a page output gets stuck emitting the same div / phrase until it
    hits max_tokens.
    """
    if not text:
        return False
    for seq_len in range(1, window_size // 2 + 1):
        candidate = text[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeats = 0
        pos = len(text) - seq_len
        while pos >= 0 and text[pos : pos + seq_len] == candidate:
            repeats += 1
            pos -= seq_len
        if repeats > max_repeats:
            return True
    return False


class RecognitionPredictor:
    """Per-block OCR. Construct with a SuryaInferenceManager (or rely on default)."""

    def __init__(self, manager: Optional[SuryaInferenceManager] = None):
        self.manager = manager
        self._disable_tqdm = settings.DISABLE_TQDM

    @property
    def disable_tqdm(self) -> bool:
        return self._disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)

    def to(self, *args, **kwargs):
        return

    def __call__(
        self,
        images: List[Image.Image],
        layout_results: Optional[List[LayoutResult]] = None,
        *,
        full_page: Optional[bool] = None,
    ) -> List[PageOCRResult]:
        """Run OCR on each page.

        Mode resolution:
          - ``full_page=None`` (default): block mode if ``layout_results`` is
            given, else full-page mode. This is the most-do-what-I-mean form.
          - ``full_page=True``: full-page OCR (single HIGH_ACCURACY_BBOX_PROMPT
            request per page). ``layout_results`` is ignored — a warning is
            logged if it was supplied.
          - ``full_page=False``: block mode (per-layout-block OCR request).
            ``layout_results`` is required.

        Full-page is the more accurate path; block mode is for callers that
        specifically need per-block crops (e.g. for downstream merging with
        text-line detection).
        """
        if not images:
            return []
        if full_page is None:
            full_page = layout_results is None
        if full_page:
            if layout_results is not None:
                logger.info(
                    "RecognitionPredictor called with full_page=True and "
                    "layout_results; layout will be used as fallback if the "
                    "full-page output devolves into a repetition loop."
                )
            return self._full_page_ocr(images, fallback_layout=layout_results)
        if layout_results is None:
            raise ValueError("layout_results required when full_page=False")
        if len(images) != len(layout_results):
            raise ValueError(
                f"images and layout_results must be same length "
                f"({len(images)} vs {len(layout_results)})"
            )
        manager = self.manager or get_default_manager()

        # Build a flat batch across all pages for max concurrency
        batch: List[BatchInputItem] = []
        block_index_map: List[tuple[int, int]] = []  # (page_idx, block_idx)
        skipped_flags: List[bool] = []

        for page_idx, (img, layout) in enumerate(zip(images, layout_results)):
            for block_idx, box in enumerate(layout.bboxes):
                skip = box.label in SKIP_CANON_LABELS
                skipped_flags.append(skip)
                if skip:
                    continue
                crop = _crop_block(img, box.polygon)
                max_tokens = image_token_budget(
                    box.count, ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING
                )
                batch.append(
                    BatchInputItem(
                        image=crop,
                        prompt_type=PROMPT_TYPE_BLOCK,
                        max_tokens=max_tokens,
                        metadata={"page_idx": page_idx, "block_idx": block_idx},
                    )
                )
                block_index_map.append((page_idx, block_idx))

        outputs = manager.generate(batch) if batch else []

        # Index outputs by (page_idx, block_idx)
        out_by_key = {}
        for out in outputs:
            key = (out.metadata["page_idx"], out.metadata["block_idx"])
            out_by_key[key] = out

        # Assemble PageOCRResult per page
        results: List[PageOCRResult] = []
        for page_idx, (img, layout) in enumerate(zip(images, layout_results)):
            w, h = img.size
            blocks: List[BlockOCRResult] = []
            for block_idx, box in enumerate(layout.bboxes):
                skip = box.label in SKIP_CANON_LABELS
                if skip:
                    blocks.append(
                        BlockOCRResult(
                            polygon=box.polygon,
                            label=box.label,
                            raw_label=box.raw_label,
                            reading_order=box.position,
                            html="",
                            skipped=True,
                            confidence=1.0,
                        )
                    )
                    continue
                out = out_by_key.get((page_idx, block_idx))
                if out is None or out.error:
                    blocks.append(
                        BlockOCRResult(
                            polygon=box.polygon,
                            label=box.label,
                            raw_label=box.raw_label,
                            reading_order=box.position,
                            html="",
                            skipped=False,
                            error=True,
                            confidence=0.0,
                        )
                    )
                    continue
                html = clean_block_html(out.raw)
                conf = out.mean_token_prob if out.mean_token_prob is not None else 1.0
                capped = (
                    getattr(out, "finish_reason", None) == "length"
                )  # cut by the token cap: partial html, flagged
                blocks.append(
                    BlockOCRResult(
                        polygon=box.polygon,
                        label=box.label,
                        raw_label=box.raw_label,
                        reading_order=box.position,
                        html=html,
                        skipped=False,
                        error=capped,
                        confidence=conf,
                        raw_logprobs=out.logprobs,
                    )
                )
            results.append(
                PageOCRResult(blocks=blocks, image_bbox=[0, 0, float(w), float(h)])
            )
        return results

    def _full_page_ocr(
        self,
        images: List[Image.Image],
        fallback_layout: Optional[List[LayoutResult]] = None,
    ) -> List[PageOCRResult]:
        """One HIGH_ACCURACY_BBOX_PROMPT request per page; parses divs into blocks.

        On per-page failure (parse error, empty output, or a detected
        repetition loop in the decoder output), falls back to layout +
        block-mode OCR for that page only. ``fallback_layout``, if given,
        provides per-page LayoutResults to use on fallback; otherwise the
        LayoutPredictor is invoked lazily for just the affected pages.
        """
        manager = self.manager or get_default_manager()
        results: List[Optional[PageOCRResult]] = [None] * len(images)

        def _build_page(out, img):
            """A good full-page output -> PageOCRResult; otherwise None (regen/fallback).
            A genuinely blank page is a valid empty result (not a failure)."""
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out is None or out.error:
                return None
            if getattr(out, "finish_reason", None) == "length":
                return None  # cut by the token cap: never accept a partial page, go to regen / block fallback
            if not out.raw:
                return (
                    PageOCRResult(blocks=[], image_bbox=page_bbox)
                    if is_blank_region(img)
                    else None
                )
            if _detect_repeat_loop(out.raw):
                return None
            try:
                parsed = parse_full_page_html(out.raw)
            except Exception:
                return None
            if not parsed and not is_blank_region(img):
                return None  # text on the page but nothing parsed (malformed header): regen / block fallback
            if (
                parsed
                and parsed[-1].truncated
                and getattr(out, "finish_reason", None) != "stop"
            ):
                return (
                    None  # visibly cut off and the server did not report a clean stop
                )
            confidence = out.mean_token_prob if out.mean_token_prob is not None else 1.0
            blocks: List[BlockOCRResult] = []
            for idx, item in enumerate(parsed):
                x0 = item.bbox[0] / settings.BBOX_SCALE * w
                y0 = item.bbox[1] / settings.BBOX_SCALE * h
                x1 = item.bbox[2] / settings.BBOX_SCALE * w
                y1 = item.bbox[3] / settings.BBOX_SCALE * h
                polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                canon = LAYOUT_PRED_RELABEL.get(item.label, item.label)
                skipped = canon in SKIP_CANON_LABELS
                blocks.append(
                    BlockOCRResult(
                        polygon=polygon,
                        label=canon,
                        raw_label=item.label,
                        reading_order=idx,
                        html="" if skipped else item.html,
                        skipped=skipped,
                        error=False,
                        confidence=confidence,
                    )
                )
            return PageOCRResult(
                blocks=_drop_blank_text_blocks(img, blocks), image_bbox=page_bbox
            )

        # Progressive-temperature regeneration (chandra-style), gated by
        # settings.SURYA_FULLPAGE_REGEN (default off -> single greedy pass, then
        # block-mode fallback, i.e. surya's prior behavior).
        rounds = _REGEN_ROUNDS if settings.SURYA_FULLPAGE_REGEN else [(0.0, None)]
        pending = list(range(len(images)))
        for round_i, (temp, top_p) in enumerate(rounds):
            if not pending:
                break
            batch = [
                BatchInputItem(
                    image=images[i],
                    prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
                    max_tokens=settings.SURYA_MAX_TOKENS_FULL_PAGE,
                    temperature=(temp if round_i > 0 else None),
                    top_p=top_p,
                    metadata={"page_idx": i},
                )
                for i in pending
            ]
            out_by_page = {o.metadata["page_idx"]: o for o in manager.generate(batch)}
            still: List[int] = []
            for i in pending:
                r = _build_page(out_by_page.get(i), images[i])
                if r is not None:
                    results[i] = r
                else:
                    still.append(i)
            if still and round_i < len(rounds) - 1:
                logger.info(
                    f"regenerating {len(still)} full-page output(s) at higher temperature"
                )
            pending = still
        needs_fallback: List[int] = pending

        # Block-mode fallback for any pages whose full-page output failed or looped.
        if needs_fallback:
            fb_images = [images[i] for i in needs_fallback]
            if fallback_layout is not None:
                fb_layouts = [fallback_layout[i] for i in needs_fallback]
            else:
                # Lazy import to avoid the surya.layout ↔ surya.recognition cycle.
                from surya.layout import LayoutPredictor

                logger.info(
                    f"running layout for {len(fb_images)} page(s) requiring "
                    f"block-mode fallback"
                )
                fb_layouts = LayoutPredictor(self.manager)(fb_images)
            fb_results = self.__call__(fb_images, fb_layouts, full_page=False)
            for fb_idx, page_idx in enumerate(needs_fallback):
                results[page_idx] = fb_results[fb_idx]

        # Backfill any still-None pages with empty results (defensive — shouldn't happen).
        out_results: List[PageOCRResult] = []
        for page_idx, img in enumerate(images):
            r = results[page_idx]
            if r is None:
                w, h = img.size
                r = PageOCRResult(blocks=[], image_bbox=[0, 0, float(w), float(h)])
            out_results.append(r)
        return out_results
