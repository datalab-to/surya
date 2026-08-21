"""End-to-end pipeline: PDF -> one merged-hierarchy, figure-captioned AST JSON.

    uv run python -m dpr_pipeline.process_pdf path/to/document.pdf

Runs OCR, builds the section/subsection hierarchy (with page-break content
merged back together — split tables, lists, paragraphs, and TOCs all become
one node instead of two), and fills in real descriptions for every
figure/picture/diagram — all in a single command, writing everything under
one folder per PDF:

    output/<pdf name>/
        ast.json        <- the one file to actually use: full hierarchy,
                            page-break-merged, figures described.
        results.json    <- raw surya OCR output, kept for traceability
                            (e.g. the AST+PDF viewer artifacts need the
                            original bboxes to draw overlays) — not
                            something you need to open yourself.
        tables/
            tables.xlsx      <- every table, one sheet each
            <caption>.csv    <- same tables, one CSV each
        review_queue.json   <- low-confidence pages, failed captions,
                                heading-numbering gaps — what to check by hand
        parameters.json     <- equation results, keyword-tagged numeric
                                mentions, and table rows as field:value
                                records — engineering values as data, not
                                prose you have to re-read to find them
        references.json     <- "as shown in Table 4-10" style mentions
                                resolved to the actual node they name, where
                                a matching caption exists in this document

Each step is also usable standalone if you only need part of this:
    build_ast.py           results.json -> ast.json (hierarchy only)
    caption_figures.py      ast.json + pdf -> ast.json (adds figure descriptions)
    export_tables.py        ast.json -> tables/ (CSV + Excel)
    review_queue.py         ast.json -> review_queue.json (what needs a human look)
    extract_parameters.py   ast.json -> parameters.json (structured values)
    resolve_references.py   ast.json -> references.json (Table/Figure mentions linked)
Each is run standalone as `uv run python -m dpr_pipeline.<module> ...`.
"""

import argparse
import json
import os
import time
from collections import defaultdict

from surya.inference import SuryaInferenceManager
from surya.logging import configure_logging, get_logger
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader
from surya.settings import settings

from . import cache
from .build_ast import build_document_tree
from .caption_figures import DEFAULT_PROMPT, caption_figures
from .export_tables import export_tables
from .extract_parameters import extract_parameters
from .resolve_references import resolve_references
from .review_queue import build_review_queue

configure_logging()
logger = get_logger()

OCR_CACHE_NAMESPACE = "ocr"


def _ocr_cache_key(pdf_path: str, page_range) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    # Model checkpoint is part of the key so switching backends/models can't
    # silently serve OCR results from a different model.
    config_bytes = f"{settings.SURYA_MODEL_CHECKPOINT}|{settings.SURYA_INFERENCE_BACKEND}|{page_range}".encode()
    return cache.hash_bytes(pdf_bytes, config_bytes)


def run_ocr(pdf_path: str, output_dir: str, page_range=None, use_cache: bool = True) -> dict:
    """Same full-page OCR path as `surya_ocr`, returned in-memory instead of
    written to disk by the CLI, so this can feed straight into the AST build.

    Cached by (pdf file bytes, page range, model checkpoint/backend) — OCR is
    slow (minutes per page on CPU) and a pure function of those inputs, so a
    second run against the same file is a cache hit instead of a re-run.
    """
    cache_key = _ocr_cache_key(pdf_path, page_range)
    if use_cache:
        cached = cache.get(OCR_CACHE_NAMESPACE, cache_key)
        if cached is not None:
            logger.info(f"OCR cache hit for {pdf_path} — skipping OCR")
            return cached

    loader = CLILoader(pdf_path, {"page_range": page_range, "output_dir": output_dir}, highres=True)

    manager = SuryaInferenceManager()
    rec_predictor = RecognitionPredictor(manager)

    start = time.time()
    page_results = rec_predictor(loader.highres_images, full_page=True)
    logger.info(f"OCR took {time.time() - start:.2f}s for {len(loader.highres_images)} page(s)")

    out_preds = defaultdict(list)
    for name, page in zip(loader.names, page_results):
        out_pred = page.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    results = dict(out_preds)
    if use_cache:
        cache.set(OCR_CACHE_NAMESPACE, cache_key, results)
    return results


def process_pdf(
    pdf_path: str,
    output_dir: str = "output",
    page_range=None,
    caption_prompt: str = DEFAULT_PROMPT,
    use_cache: bool = True,
):
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_output_dir = os.path.join(output_dir, pdf_stem)
    os.makedirs(doc_output_dir, exist_ok=True)

    logger.info(f"[1/7] Running OCR on {pdf_path}")
    results = run_ocr(pdf_path, output_dir, page_range=page_range, use_cache=use_cache)
    doc_name = next(iter(results))

    results_path = os.path.join(doc_output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    logger.info("[2/7] Building hierarchy (sections, page-break merges)")
    tree = build_document_tree(results, doc_name)

    logger.info("[3/7] Describing figures/pictures/diagrams")
    caption_figures(tree, pdf_path, caption_prompt, use_cache=use_cache)

    logger.info("[4/7] Exporting tables to CSV/Excel")
    tables_dir = os.path.join(doc_output_dir, "tables")
    written_tables = export_tables(tree, tables_dir)
    logger.info(f"Exported {len(written_tables)} table(s) to {tables_dir}")

    logger.info("[5/7] Building review queue")
    review = build_review_queue(tree)
    review_path = os.path.join(doc_output_dir, "review_queue.json")
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    logger.info(f"Flagged {sum(len(v) for v in review.values())} item(s) for review")

    logger.info("[6/7] Extracting structured parameters")
    params = extract_parameters(tree)
    params_path = os.path.join(doc_output_dir, "parameters.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    logger.info(f"Extracted {sum(len(v) for v in params.values())} parameter(s)")

    logger.info("[7/7] Resolving Table/Figure cross-references")
    references = resolve_references(tree)
    references_path = os.path.join(doc_output_dir, "references.json")
    with open(references_path, "w", encoding="utf-8") as f:
        json.dump(references, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Found {len(references['references'])} reference(s), "
        f"{len(references['references']) - references['unresolved_count']} resolved"
    )

    ast_path = os.path.join(doc_output_dir, "ast.json")
    with open(ast_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Wrote {ast_path}")
    return ast_path


def main():
    parser = argparse.ArgumentParser(description="PDF -> one hierarchy+figure-captioned AST JSON")
    parser.add_argument("pdf_path")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--page-range", default=None, help="e.g. 0,5-10,20")
    parser.add_argument("--caption-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-cache", action="store_true", help="Force re-OCR and re-caption instead of using cached results")
    args = parser.parse_args()

    process_pdf(args.pdf_path, args.output_dir, args.page_range, args.caption_prompt, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
