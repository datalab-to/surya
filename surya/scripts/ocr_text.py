import os
import click
import json
import time
from collections import defaultdict

from surya.common.surya.schema import TaskNames
from surya.detection import DetectionPredictor
from surya.debug.text import draw_text_on_image
from surya.logging import configure_logging, get_logger
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader

configure_logging()
logger = get_logger()


@click.command(help="OCR text.")
@click.option("--task_name", type=str, default=TaskNames.ocr_with_boxes)
@click.option(
    "--disable_math", is_flag=True, default=False, help="Do not recognize math in OCR."
)
@CLILoader.common_options
def ocr_text_cli(input_path: str, task_name: str, disable_math: bool, **kwargs):
    loader = CLILoader(input_path, kwargs, highres=True)
    task_names = [task_name] * len(loader.images)

    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    start = time.time()
    predictions_by_image = rec_predictor(
        loader.images,
        task_names=task_names,
        det_predictor=det_predictor,
        highres_images=loader.highres_images,
        math_mode=not disable_math,
    )

    if loader.debug:
        logger.debug(f"OCR took {time.time() - start:.2f} seconds")
        max_chars = max(
            [len(line.text) for p in predictions_by_image for line in p.text_lines]
        )
        logger.debug(f"Max chars: {max_chars}")

    if loader.save_images:
        for idx, (name, image, pred) in enumerate(
            zip(loader.names, loader.images, predictions_by_image)
        ):
            bboxes = [line.bbox for line in pred.text_lines]
            pred_text = [line.text for line in pred.text_lines]
            page_image = draw_text_on_image(bboxes, pred_text, image.size)
            page_image.save(os.path.join(loader.result_path, f"{name}_{idx}_text.png"))

    out_preds = defaultdict(list)
    for name, pred, image in zip(loader.names, predictions_by_image, loader.images):
        out_pred = pred.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(
        os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(out_preds, f, ensure_ascii=False)

    logger.info(f"Wrote results to {loader.result_path}")
