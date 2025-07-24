import re
import unicodedata
from collections import defaultdict

import click

from benchmark.utils.scoring import overlap_score, overlap_score_exact
from surya.input.processing import convert_if_not_rgb
from surya.debug.text import draw_text_on_image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings
from surya.recognition.languages import CODE_TO_LANGUAGE
from benchmark.utils.tesseract import (
    tesseract_ocr_parallel,
    surya_lang_to_tesseract,
    TESS_CODE_TO_LANGUAGE,
)
from benchmark.utils.textract import textract_ocr_parallel
import os
import datasets
import json
import time
from tabulate import tabulate

KEY_LANGUAGES = [
    "Chinese",
    "Spanish",
    "English",
    "Arabic",
    "Hindi",
    "Bengali",
    "Russian",
    "Japanese",
]


def list_in(lst: str | list, lst2: list):
    if isinstance(lst, str):
        lst = [lst]
    return any([item in lst for item in lst2])


def standardize_bullets(text):
    patterns = [
        r"•\s+",
        r"·\s+",
        r"○\s+",
        r"◦\s+",
        r"▪\s+",
        r"▫\s+",
        r"➢\s+",
        r"➤\s+",
        r"★\s+",
        r"✓\s+",
        r"✗\s+",
        r"✦\s+",
        r"\\bullet\s+",
    ]

    combined_pattern = "|".join(patterns)
    text = re.sub(combined_pattern, "*", text)

    return text


def normalize_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove LaTeX tags
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = standardize_bullets(text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip().lower().replace(",", ".")


@click.command(help="Benchmark recognition model.")
@click.option(
    "--results_dir",
    type=str,
    help="Path to JSON file with OCR results.",
    default=os.path.join(settings.RESULT_DIR, "benchmark"),
)
@click.option(
    "--max_rows", type=int, help="Maximum number of pdf pages to OCR.", default=None
)
@click.option("--debug", is_flag=True, help="Enable debug mode.", default=False)
@click.option(
    "--tesseract", is_flag=True, help="Run benchmarks on tesseract.", default=False
)
@click.option(
    "--textract", is_flag=True, help="Run benchmarks on textract.", default=False
)
@click.option(
    "--tess_cpus", type=int, help="Number of CPUs to use for tesseract.", default=28
)
@click.option(
    "--textract_cpus", type=int, help="Number of CPUs to use for textract.", default=28
)
@click.option(
    "--languages",
    type=str,
    help="Comma-separated list of languages to benchmark.",
    default=None,
)
def main(
    results_dir: str,
    max_rows: int,
    debug: bool,
    tesseract: bool,
    textract: bool,
    tess_cpus: int,
    textract_cpus: int,
    languages: str | None,
):
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    split = "train"
    dataset = datasets.load_dataset(
        settings.RECOGNITION_BENCH_DATASET_NAME, split=split
    )

    if languages:
        languages = languages.split(",")
        dataset = dataset.filter(
            lambda x: list_in(x["language"], languages), num_proc=4
        )

    if max_rows and max_rows < len(dataset):
        dataset = dataset.shuffle(seed=1).select(range(max_rows))

    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    bboxes = list(dataset["bboxes"])
    line_text = list(dataset["text"])
    languages = list(dataset["language"])

    print(f"Loaded {len(images)} images. Running OCR...")

    start = time.time()
    predictions_by_image = rec_predictor(images, None, bboxes=bboxes)
    surya_time = time.time() - start

    lang_list = []
    for lang in languages:
        if not isinstance(lang, list):
            lang_list.append([lang])
        else:
            lang_list.append(lang)

    surya_scores = defaultdict(list)
    img_surya_scores = []
    outputs = []
    for idx, (pred, ref_text, langs) in enumerate(
        zip(predictions_by_image, line_text, lang_list)
    ):
        pred_text = [line.text for line in pred.text_lines]

        score_ref_text = [normalize_text(line) for line in ref_text]
        score_pred_text = [normalize_text(text) for text in pred_text]
        image_scores, image_weights = overlap_score_exact(
            score_pred_text, score_ref_text
        )
        normalized_scores = [
            score / max(1, weight) for score, weight in zip(image_scores, image_weights)
        ]
        image_score = sum(image_scores) / max(1, sum(image_weights))

        img_surya_scores.append(image_score)
        for lang in langs:
            surya_scores[CODE_TO_LANGUAGE[lang]].append(image_score)

        assert len(pred_text) == len(ref_text) == len(bboxes[idx])
        if debug:
            for j, (pred_line, ref_line, score, bbox) in enumerate(
                zip(pred_text, ref_text, normalized_scores, bboxes[idx])
            ):
                image_slice = images[idx].crop(bbox)

                outputs.append(
                    {
                        "image": image_slice,
                        "bbox": bbox,
                        "score": score,
                        "pred": pred_line,
                        "ref": ref_line,
                        "langs": ",".join(langs),
                    }
                )

    if debug:
        out_ds = datasets.Dataset.from_list(outputs)
        out_ds.push_to_hub("datalab-to/rec_bench_outputs", private=True)

    flat_surya_scores = [score for lang in surya_scores for score in surya_scores[lang]]
    benchmark_stats = {
        "surya": {
            "avg_score": sum(flat_surya_scores) / max(1, len(flat_surya_scores)),
            "lang_scores": {
                lang: sum(scores) / max(1, len(scores))
                for lang, scores in surya_scores.items()
            },
            "time_per_img": surya_time / max(1, len(images)),
        }
    }

    result_path = os.path.join(results_dir, "rec_bench")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "surya_scores.json"), "w+") as f:
        json.dump(surya_scores, f)

    if tesseract:
        tess_valid = []
        tess_langs = []
        for idx, lang in enumerate(lang_list):
            # Tesseract does not support all languages
            tess_lang = surya_lang_to_tesseract(lang[0])
            if tess_lang is None:
                continue

            tess_valid.append(idx)
            tess_langs.append(tess_lang)

        tess_imgs = [images[i] for i in tess_valid]
        tess_bboxes = [bboxes[i] for i in tess_valid]
        tess_reference = [line_text[i] for i in tess_valid]
        start = time.time()
        tess_predictions = tesseract_ocr_parallel(
            tess_imgs, tess_bboxes, tess_langs, cpus=tess_cpus
        )
        tesseract_time = time.time() - start

        tess_scores = defaultdict(list)
        for idx, (pred, ref_text, lang) in enumerate(
            zip(tess_predictions, tess_reference, tess_langs)
        ):
            image_scores, image_weights, _ = overlap_score(pred, ref_text)
            image_score = sum(image_scores) / max(1, sum(image_weights))
            tess_scores[TESS_CODE_TO_LANGUAGE[lang]].append(image_score)

        flat_tess_scores = [
            score for lang in tess_scores for score in tess_scores[lang]
        ]
        benchmark_stats["tesseract"] = {
            "avg_score": sum(flat_tess_scores) / len(flat_tess_scores),
            "lang_scores": {
                lang: sum(scores) / len(scores) for lang, scores in tess_scores.items()
            },
            "time_per_img": tesseract_time / len(tess_imgs),
        }

        with open(os.path.join(result_path, "tesseract_scores.json"), "w+") as f:
            json.dump(tess_scores, f)

    if textract:
        start = time.time()
        textract_predictions = textract_ocr_parallel(images, cpus=textract_cpus)
        textract_time = time.time() - start

        textract_scores = defaultdict(list)
        for idx, (pred, ref_text, lang) in enumerate(
            zip(textract_predictions, line_text, lang_list)
        ):
            image_scores, image_weights, _ = overlap_score(pred, ref_text)
            image_score = sum(image_scores) / max(1, sum(image_weights))

            for lang in lang:
                textract_scores[CODE_TO_LANGUAGE[lang]].append(image_score)

        flat_textract_scores = [
            score for lang in textract_scores for score in textract_scores[lang]
        ]
        benchmark_stats["textract"] = {
            "avg_score": sum(flat_textract_scores) / len(flat_textract_scores),
            "lang_scores": {
                lang: sum(scores) / len(scores)
                for lang, scores in textract_scores.items()
            },
            "time_per_img": textract_time / len(images),
        }
        print(len(flat_textract_scores))

        with open(os.path.join(result_path, "textract_scores.json"), "w+") as f:
            json.dump(textract_scores, f)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(benchmark_stats, f)

    key_languages = [k for k in KEY_LANGUAGES if k in surya_scores]
    table_headers = ["Model", "Time per page (s)", "Avg Score"] + key_languages
    table_data = [
        [
            "surya",
            benchmark_stats["surya"]["time_per_img"],
            benchmark_stats["surya"]["avg_score"],
        ]
        + [benchmark_stats["surya"]["lang_scores"][lang] for lang in key_languages],
    ]
    if tesseract:
        table_data.append(
            [
                "tesseract",
                benchmark_stats["tesseract"]["time_per_img"],
                benchmark_stats["tesseract"]["avg_score"],
            ]
            + [
                benchmark_stats["tesseract"]["lang_scores"].get(lang, 0)
                for lang in key_languages
            ]
        )
    if textract:
        table_data.append(
            [
                "textract",
                benchmark_stats["textract"]["time_per_img"],
                benchmark_stats["textract"]["avg_score"],
            ]
            + [
                benchmark_stats["textract"]["lang_scores"][lang]
                for lang in key_languages
            ],
        )

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print(
        "Only a few major languages are displayed. See the result path for additional languages."
    )

    if debug >= 1:
        bad_detections = []
        for idx, (score, lang) in enumerate(zip(flat_surya_scores, lang_list)):
            if score < 0.8:
                bad_detections.append((idx, lang, score))
        print(f"Found {len(bad_detections)} bad detections. Writing to file...")
        with open(os.path.join(result_path, "bad_detections.json"), "w+") as f:
            json.dump(bad_detections, f)

    if debug == 2:
        for idx, (image, pred, ref_text, bbox, lang) in enumerate(
            zip(images, predictions_by_image, line_text, bboxes, lang_list)
        ):
            pred_image_name = f"{'_'.join(lang)}_{idx}_pred.png"
            ref_image_name = f"{'_'.join(lang)}_{idx}_ref.png"
            pred_text = [line.text for line in pred.text_lines]
            pred_image = draw_text_on_image(bbox, pred_text, image.size)
            pred_image.save(os.path.join(result_path, pred_image_name))
            ref_image = draw_text_on_image(bbox, ref_text, image.size)
            ref_image.save(os.path.join(result_path, ref_image_name))
            image.save(os.path.join(result_path, f"{'_'.join(lang)}_{idx}_image.png"))

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
