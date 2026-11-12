from collections import defaultdict
import os
from typing import Optional

import click

from surya.common.util import expand_bbox
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.logging import configure_logging, get_logger
from surya.scripts.config import CLILoader
from surya.table_rec import TableRecPredictor

configure_logging()
logger = get_logger()


@click.command(help="Run HTML table recognition on an input file or folder.")
@CLILoader.common_options
@click.option(
    "--skip_table_detection",
    is_flag=True,
    help="Tables are already cropped, so don't re-detect tables.",
    default=False,
)
def table_recognition_cli(
    input_path: Optional[str] = None,
    skip_table_detection: bool = False,
    **kwargs,
):
    if CLILoader.handle_skill_flags(kwargs):
        return

    # Layout runs on the low-DPI render; table crops come from the high-DPI
    # image so the table_rec model sees readable cell content.
    loader = CLILoader(input_path, kwargs, highres=True)

    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)
    table_rec_predictor = TableRecPredictor(manager)

    pnums = []
    prev_name = None
    for name in loader.names:
        if prev_name is None or prev_name != name:
            pnums.append(0)
        else:
            pnums.append(pnums[-1] + 1)
        prev_name = name

    table_imgs = []
    table_counts = []
    table_counts_per_img = []

    if skip_table_detection:
        for img in loader.highres_images:
            table_imgs.append(img)
            table_counts.append(1)
            table_counts_per_img.append(0)
    else:
        layout_predictions = layout_predictor(
            loader.images,
            target_image_sizes=[img.size for img in loader.highres_images],
        )
        for layout_pred, img in zip(layout_predictions, loader.highres_images):
            tables_on_page = [
                line
                for line in layout_pred.bboxes
                if line.label in ("Table", "TableOfContents")
            ]
            table_counts.append(len(tables_on_page))
            for line in tables_on_page:
                bbox = expand_bbox(line.bbox)
                table_imgs.append(img.crop(bbox))
                table_counts_per_img.append(line.count)

    table_preds = table_rec_predictor(table_imgs, counts=table_counts_per_img)

    img_idx = 0
    prev_count = 0
    table_predictions = defaultdict(list)
    for i in range(sum(table_counts)):
        while i >= prev_count + table_counts[img_idx]:
            prev_count += table_counts[img_idx]
            img_idx += 1

        pred = table_preds[i]
        orig_name = loader.names[img_idx]
        pnum = pnums[img_idx]
        table_img = table_imgs[i]

        out_pred = pred.model_dump()
        out_pred["page"] = pnum + 1
        table_idx = i - prev_count
        out_pred["table_idx"] = table_idx
        table_predictions[orig_name].append(out_pred)

        if loader.save_images:
            table_img.save(
                os.path.join(
                    loader.result_path,
                    f"{orig_name}_page{pnum + 1}_table{table_idx}.png",
                )
            )

    loader.write_output(table_predictions)
