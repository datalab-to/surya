import click
import collections
import json

from surya.debug.draw import draw_bboxes_on_image
from tabulate import tabulate

from surya.input.processing import convert_if_not_rgb
from surya.table_rec import TableRecPredictor
from surya.settings import settings
from benchmark.utils.metrics import penalized_iou_score
from benchmark.utils.tatr import load_tatr, batch_inference_tatr
import os
import time
import datasets


@click.command(help="Benchmark table rec dataset")
@click.option(
    "--results_dir",
    type=str,
    help="Path to JSON file with benchmark results.",
    default=os.path.join(settings.RESULT_DIR, "benchmark"),
)
@click.option(
    "--max_rows",
    type=int,
    help="Maximum number of images to run benchmark on.",
    default=512,
)
@click.option("--tatr", is_flag=True, help="Run table transformer.", default=False)
@click.option("--debug", is_flag=True, help="Enable debug mode.", default=False)
def main(results_dir: str, max_rows: int, tatr: bool, debug: bool):
    table_rec_predictor = TableRecPredictor()

    pathname = "table_rec_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    split = "train"
    if max_rows is not None:
        split = f"train[:{max_rows}]"
    dataset = datasets.load_dataset(settings.TABLE_REC_BENCH_DATASET_NAME, split=split)
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)

    if settings.TABLE_REC_STATIC_CACHE:
        # Run through one batch to compile the model
        table_rec_predictor(images[:1])

    start = time.time()
    table_rec_predictions = table_rec_predictor(images)
    surya_time = time.time() - start

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    page_metrics = collections.OrderedDict()
    mean_col_iou = 0
    mean_row_iou = 0
    for idx, (pred, image) in enumerate(zip(table_rec_predictions, images)):
        row = dataset[idx]
        pred_row_boxes = [p.bbox for p in pred.rows]
        pred_col_bboxes = [p.bbox for p in pred.cols]
        actual_row_bboxes = [r["bbox"] for r in row["rows"]]
        actual_col_bboxes = [c["bbox"] for c in row["columns"]]
        row_score = penalized_iou_score(pred_row_boxes, actual_row_bboxes)
        col_score = penalized_iou_score(pred_col_bboxes, actual_col_bboxes)
        page_results = {
            "row_score": row_score,
            "col_score": col_score,
            "row_count": len(actual_row_bboxes),
            "col_count": len(actual_col_bboxes),
        }

        mean_col_iou += col_score
        mean_row_iou += row_score

        page_metrics[idx] = page_results

        if debug:
            # Save debug images
            draw_img = image.copy()
            draw_bboxes_on_image(
                pred_row_boxes,
                draw_img,
                [f"Row {i}" for i in range(len(pred_row_boxes))],
            )
            draw_bboxes_on_image(
                pred_col_bboxes,
                draw_img,
                [f"Col {i}" for i in range(len(pred_col_bboxes))],
                color="blue",
            )
            draw_img.save(os.path.join(result_path, f"{idx}_bbox.png"))

            actual_draw_image = image.copy()
            draw_bboxes_on_image(
                actual_row_bboxes,
                actual_draw_image,
                [f"Row {i}" for i in range(len(actual_row_bboxes))],
            )
            draw_bboxes_on_image(
                actual_col_bboxes,
                actual_draw_image,
                [f"Col {i}" for i in range(len(actual_col_bboxes))],
                color="blue",
            )
            actual_draw_image.save(os.path.join(result_path, f"{idx}_actual.png"))

    mean_col_iou /= len(table_rec_predictions)
    mean_row_iou /= len(table_rec_predictions)

    out_data = {
        "surya": {
            "time": surya_time,
            "mean_row_iou": mean_row_iou,
            "mean_col_iou": mean_col_iou,
            "page_metrics": page_metrics,
        }
    }

    if tatr:
        tatr_model = load_tatr()
        start = time.time()
        tatr_predictions = batch_inference_tatr(tatr_model, images, 1)
        tatr_time = time.time() - start

        page_metrics = collections.OrderedDict()
        mean_col_iou = 0
        mean_row_iou = 0
        for idx, pred in enumerate(tatr_predictions):
            row = dataset[idx]
            pred_row_boxes = [p["bbox"] for p in pred["rows"]]
            pred_col_bboxes = [p["bbox"] for p in pred["cols"]]
            actual_row_bboxes = [r["bbox"] for r in row["rows"]]
            actual_col_bboxes = [c["bbox"] for c in row["columns"]]
            row_score = penalized_iou_score(pred_row_boxes, actual_row_bboxes)
            col_score = penalized_iou_score(pred_col_bboxes, actual_col_bboxes)
            page_results = {
                "row_score": row_score,
                "col_score": col_score,
                "row_count": len(actual_row_bboxes),
                "col_count": len(actual_col_bboxes),
            }

            mean_col_iou += col_score
            mean_row_iou += row_score

            page_metrics[idx] = page_results

        mean_col_iou /= len(tatr_predictions)
        mean_row_iou /= len(tatr_predictions)

        out_data["tatr"] = {
            "time": tatr_time,
            "mean_row_iou": mean_row_iou,
            "mean_col_iou": mean_col_iou,
            "page_metrics": page_metrics,
        }

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    table = [
        ["Model", "Row Intersection", "Col Intersection", "Time Per Image"],
        [
            "Surya",
            f"{out_data['surya']['mean_row_iou']:.2f}",
            f"{out_data['surya']['mean_col_iou']:.5f}",
            f"{surya_time / len(images):.5f}",
        ],
    ]

    if tatr:
        table.append(
            [
                "Table transformer",
                f"{out_data['tatr']['mean_row_iou']:.2f}",
                f"{out_data['tatr']['mean_col_iou']:.5f}",
                f"{tatr_time / len(images):.5f}",
            ]
        )

    print(tabulate(table, headers="firstrow", tablefmt="github"))

    print(
        "Intersection is the average of the intersection % between each actual row/column, and the predictions.  With penalties for too many/few predictions."
    )
    print(
        "Note that table transformers is unbatched, since the example code in the repo is unbatched."
    )
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
