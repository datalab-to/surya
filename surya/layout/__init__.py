from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.common.blank import is_blank_region
from surya.inference import SuryaInferenceManager, get_default_manager
from surya.inference.parsers import denorm_bbox, parse_layout
from surya.inference.prompts import LAYOUT_JSON_SCHEMA, PROMPT_TYPE_LAYOUT
from surya.inference.schema import BatchInputItem
from surya.layout.label import LAYOUT_PRED_RELABEL, TEXT_LABELS
from surya.layout.schema import LayoutBox, LayoutResult
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


class LayoutPredictor:
    """Run LAYOUT_PROMPT on full pages, parse JSON, return LayoutResult per image."""

    def __init__(self, manager: Optional[SuryaInferenceManager] = None):
        self.manager = manager  # If None, get_default_manager() is used at call time
        self._disable_tqdm = settings.DISABLE_TQDM

    @property
    def disable_tqdm(self) -> bool:
        return self._disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)

    def to(self, *args, **kwargs):
        # Manager-backed; .to() is a no-op for compatibility with BasePredictor callers.
        return

    def __call__(
        self,
        images: List[Image.Image],
        target_image_sizes: Optional[List[tuple]] = None,
        max_tokens: Optional[int] = None,
    ) -> List[LayoutResult]:
        """Run layout on a batch of images.

        target_image_sizes: optional list of (width, height) tuples — if
        provided, bboxes are denormalized to these sizes instead of each
        input image's size. Useful when layout runs on a low-DPI render but
        you want bboxes in the OCR image's coordinate space.
        """
        if not isinstance(images, (list, tuple)):
            raise TypeError(
                f"Expected a list of PIL.Image.Image objects, got {type(images).__name__}."
            )
        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f"Expected PIL.Image.Image at index {idx}, got {type(img).__name__}. "
                    "Load files into PIL images first (e.g. with surya.input.load.load_from_file or PIL.Image.open)."
                )
        if not images:
            return []
        manager = self.manager or get_default_manager()

        max_tokens = max_tokens or settings.SURYA_MAX_TOKENS_LAYOUT
        guided = LAYOUT_JSON_SCHEMA if settings.SURYA_GUIDED_LAYOUT else None
        batch = [
            BatchInputItem(
                image=img,
                prompt_type=PROMPT_TYPE_LAYOUT,
                max_tokens=max_tokens,
                guided_json=guided,
            )
            for img in images
        ]
        outputs = manager.generate(batch)

        if target_image_sizes is not None and len(target_image_sizes) != len(images):
            raise ValueError("target_image_sizes must match images length")

        results: List[LayoutResult] = []
        for idx, (img, out) in enumerate(zip(images, outputs)):
            if target_image_sizes is not None:
                w, h = target_image_sizes[idx]
            else:
                w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error or not out.raw:
                results.append(
                    LayoutResult(
                        bboxes=[], image_bbox=page_bbox, raw=out.raw, error=True
                    )
                )
                continue
            try:
                parsed = parse_layout(out.raw)
            except Exception as e:
                logger.warning(f"Layout parse failed: {e}; raw[:300]={out.raw[:300]!r}")
                results.append(
                    LayoutResult(
                        bboxes=[], image_bbox=page_bbox, raw=out.raw, error=True
                    )
                )
                continue

            confidence = out.mean_token_prob if out.mean_token_prob is not None else 1.0
            img_w, img_h = img.size
            boxes: List[LayoutBox] = []
            dropped_blank = 0
            for blk in parsed:
                canon = LAYOUT_PRED_RELABEL.get(blk.label, blk.label)
                # Drop text-labeled blocks the model hallucinated over an
                # essentially-blank region (mostly white OR near-uniform
                # color). Visual blocks (Picture / Figure / Table / etc.)
                # are allowed to be uniform — that's normal content.
                if canon in TEXT_LABELS:
                    img_bbox = denorm_bbox(
                        blk.bbox, img_w, img_h, scale=settings.BBOX_SCALE
                    )
                    x0, y0, x1, y1 = (max(0, int(v)) for v in img_bbox)
                    if x1 > x0 and y1 > y0:
                        if is_blank_region(img.crop((x0, y0, x1, y1))):
                            dropped_blank += 1
                            continue
                pixel_bbox = denorm_bbox(blk.bbox, w, h, scale=settings.BBOX_SCALE)
                boxes.append(
                    LayoutBox(
                        polygon=list(pixel_bbox),
                        label=canon,
                        raw_label=blk.label,
                        position=len(boxes),
                        count=blk.count,
                        confidence=confidence,
                    )
                )
            if dropped_blank:
                logger.info(
                    f"dropped {dropped_blank} text-labeled layout block(s) over "
                    f"blank/uniform regions"
                )
            results.append(
                LayoutResult(
                    bboxes=boxes, image_bbox=page_bbox, raw=out.raw, error=False
                )
            )
        return results
