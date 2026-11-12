from collections import defaultdict
import copy
import os
import time
from typing import Optional

import click

from surya.debug.draw import draw_polys_on_image
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.logging import configure_logging, get_logger
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader

configure_logging()
logger = get_logger()


@click.command(
    help="OCR text — full-page OCR (default) or layout + block OCR (--block_mode)."
)
@CLILoader.common_options
@click.option(
    "--block_mode",
    is_flag=True,
    default=False,
    help="Run layout detection first, then OCR each block crop individually.",
)
def ocr_text_cli(
    input_path: Optional[str] = None, block_mode: bool = False, **kwargs
):
    if CLILoader.handle_skill_flags(kwargs):
        return

    # Full-page OCR is the default path: one VLM call per page returns layout
    # + content together. Pages whose full-page output fails to parse fall
    # back to layout + per-block OCR automatically (see RecognitionPredictor).
    loader = CLILoader(input_path, kwargs, highres=True)

    manager = SuryaInferenceManager()
    rec_predictor = RecognitionPredictor(manager)

    start = time.time()
    if block_mode:
        layout_predictor = LayoutPredictor(manager)
        layouts = layout_predictor(
            loader.images,
            target_image_sizes=[img.size for img in loader.highres_images],
        )
        page_results = rec_predictor(
            loader.highres_images, layouts, full_page=False
        )
    else:
        page_results = rec_predictor(loader.highres_images, full_page=True)

    if loader.debug:
        logger.debug(f"OCR took {time.time() - start:.2f} seconds")

    if loader.save_images:
        for idx, (image, page, name) in enumerate(
            zip(loader.highres_images, page_results, loader.names)
        ):
            polygons = [blk.polygon for blk in page.blocks]
            labels = [f"{blk.label}-{blk.reading_order}" for blk in page.blocks]
            bbox_image = draw_polys_on_image(
                polygons, copy.deepcopy(image), labels=labels
            )
            bbox_image.save(
                os.path.join(loader.result_path, f"{name}_{idx}_blocks.png")
            )

    out_preds = defaultdict(list)
    for name, page in zip(loader.names, page_results):
        out_pred = page.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    loader.write_output(out_preds)
