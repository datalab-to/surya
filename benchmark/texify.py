import os.path
import re
import time
from pathlib import Path
from typing import List

import click
import datasets
from tabulate import tabulate
from bs4 import BeautifulSoup

from surya.common.surya.schema import TaskNames
from surya.settings import settings
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor, OCRResult
import json
from rapidfuzz.distance import Levenshtein


def normalize_text(text):
    soup = BeautifulSoup(text, "html.parser")
    # Unwrap math tags
    for tag in soup.find_all():
        if tag.name == "math":
            tag.unwrap()
    text = soup.get_text()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_text(predictions, references):
    lev_dist = []
    for p, r in zip(predictions, references):
        p = normalize_text(p)
        r = normalize_text(r)
        lev_dist.append(Levenshtein.normalized_distance(p, r))

    return sum(lev_dist) / len(lev_dist)


def inference_texify(
    source_data, predictor: RecognitionPredictor, line_mode: bool = False
):
    images = [sd["image"] for sd in source_data]
    mode = TaskNames.ocr_with_boxes if line_mode else TaskNames.block_without_boxes
    tasks = [mode] * len(images)
    bboxes = [[[0, 0, image.width, image.height]] for image in images]
    texify_predictions: List[OCRResult] = predictor(images, tasks, bboxes=bboxes)
    out_data = [
        {
            "text": texify_predictions[i].text_lines[0].text,
            "equation": source_data[i]["equation"],
        }
        for i in range(len(texify_predictions))
    ]

    return out_data


@click.command(help="Benchmark the performance of texify.")
@click.option(
    "--ds_name",
    type=str,
    help="Path to dataset file with source images/equations.",
    default=settings.TEXIFY_BENCHMARK_DATASET,
)
@click.option(
    "--results_dir",
    type=str,
    help="Path to JSON file with benchmark results.",
    default=os.path.join(settings.RESULT_DIR, "benchmark"),
)
@click.option(
    "--max_rows", type=int, help="Maximum number of images to benchmark.", default=None
)
@click.option(
    "--line_mode", is_flag=True, help="Use line mode for texify.", default=False
)
def main(ds_name: str, results_dir: str, max_rows: int, line_mode: bool):
    foundation_predictor = FoundationPredictor()
    predictor = RecognitionPredictor(foundation_predictor)
    ds = datasets.load_dataset(ds_name, split="train")

    if max_rows:
        ds = ds.filter(lambda x, idx: idx < max_rows, with_indices=True)

    start = time.time()
    predictions = inference_texify(ds, predictor, line_mode)
    time_taken = time.time() - start

    text = [p["text"] for p in predictions]
    references = [p["equation"] for p in predictions]
    scores = score_text(text, references)

    write_data = {
        "scores": scores,
        "text": [{"prediction": p, "reference": r} for p, r in zip(text, references)],
    }

    score_table = [["texify", write_data["scores"], time_taken]]
    score_headers = ["edit", "time taken (s)"]
    score_dirs = ["⬇", "⬇"]

    score_headers = [f"{h} {d}" for h, d in zip(score_headers, score_dirs)]
    table = tabulate(score_table, headers=["Method", *score_headers])
    print()
    print(table)

    result_path = Path(results_dir) / "texify_bench"
    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(write_data, f, indent=4)


if __name__ == "__main__":
    main()
