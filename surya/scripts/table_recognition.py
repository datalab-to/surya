import os
import click
import copy
import json
from collections import defaultdict

from surya.common.util import expand_bbox
from surya.debug.draw import draw_bboxes_on_image
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.logging import configure_logging, get_logger
from surya.scripts.config import CLILoader
from surya.table_rec import TableRecPredictor

configure_logging()
logger = get_logger()


@click.command(help="Run table recognition on an input file or folder.")
@CLILoader.common_options
@click.option(
    "--skip_table_detection",
    is_flag=True,
    help="Tables are already cropped, so don't re-detect tables.",
    default=False,
)
@click.option(
    "--mode",
    type=click.Choice(["simple", "full"]),
    default="full",
    help="full: table HTML with colspan/rowspan. simple: rows+cols only (geometric cells; Surya 2 checkpoints only).",
)
def table_recognition_cli(
    input_path: str, skip_table_detection: bool, mode: str, **kwargs
):
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

    table_preds = table_rec_predictor(table_imgs, mode=mode)

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

        if loader.save_images and pred.rows:
            rows = [line.bbox for line in pred.rows]
            cols = [line.bbox for line in pred.cols]
            row_labels = [f"Row {line.row_id}" for line in pred.rows]
            col_labels = [f"Col {line.col_id}" for line in pred.cols]
            cells = [line.bbox for line in pred.cells]

            rc_image = copy.deepcopy(table_img)
            rc_image = draw_bboxes_on_image(
                rows, rc_image, labels=row_labels, label_font_size=20, color="blue"
            )
            rc_image = draw_bboxes_on_image(
                cols, rc_image, labels=col_labels, label_font_size=20, color="red"
            )
            rc_image.save(
                os.path.join(
                    loader.result_path,
                    f"{orig_name}_page{pnum + 1}_table{table_idx}_rc.png",
                )
            )

            cell_image = copy.deepcopy(table_img)
            cell_image = draw_bboxes_on_image(cells, cell_image, color="green")
            cell_image.save(
                os.path.join(
                    loader.result_path,
                    f"{orig_name}_page{pnum + 1}_table{table_idx}_cells.png",
                )
            )

    with open(
        os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(table_predictions, f, ensure_ascii=False)

    logger.info(f"Wrote results to {loader.result_path}")
