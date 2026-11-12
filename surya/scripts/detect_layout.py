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
from surya.scripts.config import CLILoader

configure_logging()
logger = get_logger()


@click.command(help="Detect layout of an input file or folder (PDFs or image).")
@CLILoader.common_options
def detect_layout_cli(input_path: Optional[str] = None, **kwargs):
    if CLILoader.handle_skill_flags(kwargs):
        return

    loader = CLILoader(input_path, kwargs)

    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)

    start = time.time()
    layout_predictions = layout_predictor(loader.images)

    if loader.debug:
        logger.debug(f"Layout took {time.time() - start:.2f} seconds")

    if loader.save_images:
        for idx, (image, layout_pred, name) in enumerate(
            zip(loader.images, layout_predictions, loader.names)
        ):
            polygons = [p.polygon for p in layout_pred.bboxes]
            labels = [f"{p.label}-{p.position}" for p in layout_pred.bboxes]
            bbox_image = draw_polys_on_image(
                polygons, copy.deepcopy(image), labels=labels
            )
            bbox_image.save(
                os.path.join(loader.result_path, f"{name}_{idx}_layout.png")
            )

    predictions_by_page = defaultdict(list)
    for pred, name in zip(layout_predictions, loader.names):
        out_pred = pred.model_dump()
        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    loader.write_output(predictions_by_page)
