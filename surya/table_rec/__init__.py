"""TableRecPredictor: HTML table structure and content recognition.

Runs `BLOCK_PROMPT` on table crops and returns `TableResult` with full `<table>`
HTML (`colspan`, `rowspan`, `<th>`, `<td>`).
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.inference import SuryaInferenceManager, get_default_manager
from surya.inference.parsers import clean_block_html
from surya.inference.prompts import PROMPT_TYPE_BLOCK
from surya.inference.schema import BatchInputItem
from surya.inference.util import image_token_budget
from surya.logging import get_logger
from surya.settings import settings
from surya.table_rec.schema import TableResult

logger = get_logger()


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
        self,
        images: List[Image.Image],
        counts: Optional[List[int]] = None,
    ) -> List[TableResult]:
        return self.predict_full(images, counts=counts)

    def predict_full(
        self, images: List[Image.Image], counts: Optional[List[int]] = None
    ) -> List[TableResult]:
        """Run BLOCK_PROMPT on table crops and return full <table> HTML."""
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
        if counts is None:
            counts = [0] * len(images)
        batch = [
            BatchInputItem(
                image=img,
                prompt_type=PROMPT_TYPE_BLOCK,
                max_tokens=image_token_budget(
                    count,
                    ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING,
                    floor=1024,
                ),
            )
            for img, count in zip(images, counts)
        ]
        outputs = manager.generate(batch)
        results: List[TableResult] = []
        for img, out in zip(images, outputs):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error:
                results.append(
                    TableResult(
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
                    image_bbox=page_bbox,
                    raw=out.raw,
                    html=html,
                    mode="full",
                    error=getattr(out, "finish_reason", None)
                    == "length",  # partial html is kept, flagged as an error
                )
            )
        return results
