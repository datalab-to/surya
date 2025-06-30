import collections
import copy
import json

import click

from benchmark.utils.metrics import precision_recall
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.input.processing import convert_if_not_rgb
from surya.debug.draw import draw_bboxes_on_image
from surya.settings import settings
import os
import time
from tabulate import tabulate
import datasets


@click.command(help="Benchmark surya layout model.")
@click.option("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
@click.option("--max_rows", type=int, help="Maximum number of images to run benchmark on.", default=100)
@click.option("--debug", is_flag=True, help="Run in debug mode.", default=False)
def main(results_dir: str, max_rows: int, debug: bool):
    foundation_predictor = FoundationPredictor()
    layout_predictor = LayoutPredictor(foundation_predictor)

    pathname = "layout_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    dataset = datasets.load_dataset(settings.LAYOUT_BENCH_DATASET_NAME, split=f"train[:{max_rows}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)

    if settings.LAYOUT_STATIC_CACHE:
        layout_predictor(images[:1])

    start = time.time()
    layout_predictions = layout_predictor(images)
    surya_time = time.time() - start

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    label_alignment = { # First is publaynet, second is surya
        "Image": [["Figure"], ["Picture", "Figure"]],
        "Table": [["Table"], ["Table", "Form", "TableOfContents"]],
        "Text": [["Text"], ["Text", "Formula", "Footnote", "Caption", "TextInlineMath", "Code", "Handwriting"]],
        "List": [["List"], ["ListItem"]],
        "Title": [["Title"], ["SectionHeader", "Title"]]
    }

    page_metrics = collections.OrderedDict()
    for idx, pred in enumerate(layout_predictions):
        row = dataset[idx]
        all_correct_bboxes = []
        page_results = {}
        for label_name in label_alignment:
            correct_cats, surya_cats = label_alignment[label_name]
            correct_bboxes = [b for b, l in zip(row["bboxes"], row["labels"]) if l in correct_cats]
            all_correct_bboxes.extend(correct_bboxes)
            pred_bboxes = [b.bbox for b in pred.bboxes if b.label in surya_cats]

            metrics = precision_recall(pred_bboxes, correct_bboxes, penalize_double=False)
            weight = len(correct_bboxes)
            metrics["weight"] = weight
            page_results[label_name] = metrics

        page_metrics[idx] = page_results

        if debug:
            bbox_image = draw_bboxes_on_image(all_correct_bboxes, copy.deepcopy(images[idx]))
            bbox_image.save(os.path.join(result_path, f"{idx}_layout.png"))

    mean_metrics = collections.defaultdict(dict)
    layout_types = sorted(page_metrics[0].keys())
    metric_types = sorted(page_metrics[0][layout_types[0]].keys())
    metric_types.remove("weight")
    for l in layout_types:
        for m in metric_types:
            metric = []
            total = 0
            for page in page_metrics:
                metric.append(page_metrics[page][l][m] * page_metrics[page][l]["weight"])
                total += page_metrics[page][l]["weight"]

            value = sum(metric)
            if value > 0:
                value /= total
            mean_metrics[l][m] = value

    out_data = {
        "time": surya_time,
        "metrics": mean_metrics,
        "page_metrics": page_metrics
    }

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Layout Type", ] + metric_types
    table_data = []
    for layout_type in layout_types:
        table_data.append([layout_type, ] + [f"{mean_metrics[layout_type][m]:.5f}" for m in metric_types])

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print(f"Took {surya_time / len(images):.5f} seconds per image, and {surya_time:.5f} seconds total.")
    print("Precision and recall are over the mutual coverage of the detected boxes and the ground truth boxes at a .5 threshold.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
