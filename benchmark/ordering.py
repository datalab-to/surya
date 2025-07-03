import collections
import json

import click

from surya.input.processing import convert_if_not_rgb
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.common.polygon import PolygonBox
from surya.settings import settings
from benchmark.utils.metrics import rank_accuracy
import os
import time
import datasets


@click.command(help="Benchmark surya layout for reading order.")
@click.option("--results_dir", type=str, help="Path to JSON file with benchmark results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
@click.option("--max_rows", type=int, help="Maximum number of images to run benchmark on.", default=None)
def main(results_dir: str, max_rows: int):
    foundation_predictor = FoundationPredictor()
    layout_predictor = LayoutPredictor(foundation_predictor)
    pathname = "order_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    split = "train"
    if max_rows is not None:
        split = f"train[:{max_rows}]"
    dataset = datasets.load_dataset(settings.ORDER_BENCH_DATASET_NAME, split=split)
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)

    start = time.time()
    layout_predictions = layout_predictor(images)
    surya_time = time.time() - start

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    page_metrics = collections.OrderedDict()
    mean_accuracy = 0
    for idx, order_pred in enumerate(layout_predictions):
        row = dataset[idx]
        labels = row["labels"]
        bboxes = row["bboxes"]
        pred_positions = []
        for label, bbox in zip(labels, bboxes):
            max_intersection = 0
            matching_idx = 0
            for pred_box in order_pred.bboxes:
                intersection = pred_box.intersection_pct(PolygonBox(polygon=bbox))
                if intersection > max_intersection:
                    max_intersection = intersection
                    matching_idx = pred_box.position
            pred_positions.append(matching_idx)
        accuracy = rank_accuracy(pred_positions, labels)
        mean_accuracy += accuracy
        page_results = {
            "accuracy": accuracy,
            "box_count": len(labels)
        }

        page_metrics[idx] = page_results

    mean_accuracy /= len(layout_predictions)

    out_data = {
        "time": surya_time,
        "mean_accuracy": mean_accuracy,
        "page_metrics": page_metrics
    }

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    print(f"Mean accuracy is {mean_accuracy:.2f}.")
    print(f"Took {surya_time / len(images):.2f} seconds per image, and {surya_time:.1f} seconds total.")
    print("Mean accuracy is the % of correct ranking pairs.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
