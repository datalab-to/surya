"""TableRecPredictor: dual-path table structure recognition.

- predict_simple: TABLE_REC_PROMPT → rows + columns only, cells derived
  geometrically (row × column intersections).
- predict_full: BLOCK_PROMPT on the table crop → full <table> HTML with
  colspan / rowspan / <th>. The HTML lives on TableResult.html for marker to
  consume directly.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.inference import SuryaInferenceManager, get_default_manager
from surya.inference.parsers import clean_block_html, denorm_bbox, parse_table_rec
from surya.inference.prompts import (
    PROMPT_TYPE_BLOCK,
    PROMPT_TYPE_TABLE_REC,
    TABLE_REC_JSON_SCHEMA,
)
from surya.inference.schema import BatchInputItem
from surya.inference.util import image_token_budget
from surya.logging import get_logger
from surya.settings import settings
from surya.table_rec.schema import TableCell, TableCol, TableResult, TableRow

logger = get_logger()


def _polygon_from_bbox(bbox):
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _intersect_bbox(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


class TableRecPredictor:
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
        self, images: List[Image.Image], mode: str = "simple"
    ) -> List[TableResult]:
        if mode == "full":
            return self.predict_full(images)
        return self.predict_simple(images)

    def predict_simple(self, images: List[Image.Image]) -> List[TableResult]:
        if not images:
            return []
        manager = self.manager or get_default_manager()
        guided = TABLE_REC_JSON_SCHEMA if settings.SURYA_GUIDED_TABLE_REC else None
        batch = [
            BatchInputItem(
                image=img,
                prompt_type=PROMPT_TYPE_TABLE_REC,
                max_tokens=settings.SURYA_MAX_TOKENS_TABLE_REC,
                guided_json=guided,
            )
            for img in images
        ]
        outputs = manager.generate(batch)

        results: List[TableResult] = []
        for img, out in zip(images, outputs):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error or not out.raw:
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="simple",
                        error=True,
                    )
                )
                continue
            try:
                elements = parse_table_rec(out.raw)
            except Exception as e:
                logger.warning(
                    f"Table rec parse failed: {e}; raw[:200]={out.raw[:200]!r}"
                )
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="simple",
                        error=True,
                    )
                )
                continue

            if not elements:
                # A model that does not know this task can answer it as another one (e.g. layout JSON);
                # parse_table_rec drops non-Row/Col entries, so surface that as an error, not an empty table.
                logger.warning(
                    "Table rec output has no Row/Col entries, so this checkpoint likely does not support row/column "
                    "table structure (Surya 2.5 does not). For row/column geometry run a Surya 2 checkpoint "
                    "(SURYA_MODEL_CHECKPOINT=datalab-to/surya-ocr-2 with its own inference server); "
                    "predict_full / --mode full still returns table HTML. "
                    f"raw[:200]={out.raw[:200]!r}"
                )
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="simple",
                        error=True,
                    )
                )
                continue

            rows: List[TableRow] = []
            cols: List[TableCol] = []
            for el in elements:
                pixel_bbox = denorm_bbox(el.bbox, w, h, scale=settings.BBOX_SCALE)
                poly = _polygon_from_bbox(pixel_bbox)
                if el.label == "Row":
                    rows.append(TableRow(polygon=poly, row_id=len(rows)))
                else:
                    cols.append(TableCol(polygon=poly, col_id=len(cols)))

            # Derive cells geometrically (row × column intersections)
            cells: List[TableCell] = []
            cell_id = 0
            for row in rows:
                for col in cols:
                    inter = _intersect_bbox(row.bbox, col.bbox)
                    if inter is None:
                        continue
                    cells.append(
                        TableCell(
                            polygon=_polygon_from_bbox(inter),
                            row_id=row.row_id,
                            col_id=col.col_id,
                            cell_id=cell_id,
                        )
                    )
                    cell_id += 1
            results.append(
                TableResult(
                    rows=rows,
                    cols=cols,
                    cells=cells,
                    image_bbox=page_bbox,
                    raw=out.raw,
                    mode="simple",
                    error=False,
                )
            )
        return results

    def predict_full(
        self, images: List[Image.Image], counts: Optional[List[int]] = None
    ) -> List[TableResult]:
        """Full-HTML path: BLOCK_PROMPT on table crops. Use when complex
        structure (spanning cells, headers) matters and ground-truth-style
        HTML is preferred. `counts` (one per image) shapes max_tokens."""
        if not images:
            return []
        manager = self.manager or get_default_manager()
        if counts is None:
            counts = [0] * len(images)
        batch = []
        for img, count in zip(images, counts):
            batch.append(
                BatchInputItem(
                    image=img,
                    prompt_type=PROMPT_TYPE_BLOCK,
                    max_tokens=image_token_budget(
                        count,
                        ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING,
                        floor=1024,
                    ),
                )
            )
        outputs = manager.generate(batch)
        results: List[TableResult] = []
        for img, out in zip(images, outputs):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error:
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="full",
                        error=True,
                    )
                )
                continue
            html = clean_block_html(out.raw)
            results.append(
                TableResult(
                    rows=[],
                    cols=[],
                    cells=[],
                    image_bbox=page_bbox,
                    raw=out.raw,
                    html=html,
                    mode="full",
                    error=getattr(out, "finish_reason", None)
                    == "length",  # partial html is kept, flagged as an error
                )
            )
        return results
