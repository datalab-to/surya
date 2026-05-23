import os
import click
import copy
import json
from collections import defaultdict

from surya.logging import configure_logging, get_logger
from surya.scripts.config import CLILoader
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor
from surya.debug.draw import draw_bboxes_on_image
from surya.common.util import rescale_bbox, expand_bbox
from surya.settings import settings

configure_logging()
logger = get_logger()


@click.command(help="Detect layout of an input file or folder (PDFs or image).")
@CLILoader.common_options
@click.option(
    "--skip_table_detection",
    is_flag=True,
    help="Tables are already cropped, so don't re-detect tables.",
    default=False,
)
def table_recognition_cli(input_path: str, skip_table_detection: bool, **kwargs):
    loader = CLILoader(input_path, kwargs, highres=True)

    foundation_predictor = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    layout_predictor = LayoutPredictor(foundation_predictor)
    table_rec_predictor = TableRecPredictor()

    pnums = []
    prev_name = None
    for i, name in enumerate(loader.names):
        if prev_name is None or prev_name != name:
            pnums.append(0)
        else:
            pnums.append(pnums[-1] + 1)

        prev_name = name

    layout_predictions = layout_predictor(loader.images)

    table_imgs = []
    table_counts = []

    for layout_pred, img, highres_img in zip(
        layout_predictions, loader.images, loader.highres_images
    ):
        # The table may already be cropped
        if skip_table_detection:
            table_imgs.append(highres_img)
            table_counts.append(1)
        else:
            # The bbox for the entire table
            bbox = [
                line.bbox
                for line in layout_pred.bboxes
                if line.label in ["Table", "TableOfContents"]
            ]
            # Number of tables per page
            table_counts.append(len(bbox))

            if len(bbox) == 0:
                continue

            page_table_imgs = []
            highres_bbox = []
            for bb in bbox:
                highres_bb = rescale_bbox(bb, img.size, highres_img.size)
                highres_bb = expand_bbox(highres_bb)
                page_table_imgs.append(highres_img.crop(highres_bb))
                highres_bbox.append(highres_bb)

            table_imgs.extend(page_table_imgs)

    table_preds = table_rec_predictor(table_imgs)

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
                    loader.result_path, f"{name}_page{pnum + 1}_table{table_idx}_rc.png"
                )
            )

            cell_image = copy.deepcopy(table_img)
            cell_image = draw_bboxes_on_image(cells, cell_image, color="green")
            cell_image.save(
                os.path.join(
                    loader.result_path,
                    f"{name}_page{pnum + 1}_table{table_idx}_cells.png",
                )
            )

    with open(
        os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(table_predictions, f, ensure_ascii=False)

    logger.info(f"Wrote results to {loader.result_path}")
