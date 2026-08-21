"""Stitch multiple PDF excerpts of one larger document into a single,
continuous AST — because a real DPR usually arrives as several page-range
exports (e.g. "-15-16.pdf", "-46-47.pdf", "-75-76.pdf") rather than one file.

Usage:
    uv run python -m dpr_pipeline.process_batch doc_name file1.pdf file2.pdf file3.pdf ...

Files are stitched in the order given on the command line — there's no
reliable way to infer "this excerpt comes before that one" from filenames
alone, so the caller states the order explicitly. Pages are renumbered
globally and sequentially across all files (file1's pages 1..N, then
file2's pages 1..M as global pages N+1..N+M, etc.), and every node also
keeps source_file + source_page_local so anything that needs the actual
PDF pixels (figure captioning) can still find them.

Runs the exact same build_document_tree() used for a single file — section
continuity, table/list/paragraph page-break merging, and numbering
correction all work identically across a *file* boundary as they already do
across a plain *page* boundary, since none of that logic is page-number- or
file-aware to begin with (see build_ast.py's module docstring).

Writes output/<doc_name>/ with the same file layout as process_pdf.py.
"""

import argparse
import json
import os

from surya.logging import configure_logging, get_logger

from .build_ast import build_document_tree
from .caption_figures import DEFAULT_PROMPT, caption_figures
from .export_tables import export_tables
from .extract_parameters import extract_parameters
from .process_pdf import run_ocr
from .resolve_references import resolve_references
from .review_queue import build_review_queue

configure_logging()
logger = get_logger()


def stitch_results(pdf_paths: list, output_dir: str, use_cache: bool = True) -> tuple:
    """Run OCR on each file (cached) and combine into one page-renumbered
    results dict, keyed under a synthetic single document name."""
    combined_pages = []
    global_page = 0
    pdf_map = {}

    for pdf_path in pdf_paths:
        pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_map[pdf_stem] = pdf_path

        logger.info(f"OCR: {pdf_path}")
        results = run_ocr(pdf_path, output_dir, page_range=None, use_cache=use_cache)
        pages = next(iter(results.values()))

        for page in pages:
            global_page += 1
            local_page = page["page"]
            new_page = dict(page)
            new_page["page"] = global_page
            new_page["blocks"] = [
                {**block, "source_file": pdf_stem, "source_page_local": local_page}
                for block in page["blocks"]
            ]
            combined_pages.append(new_page)

    return combined_pages, pdf_map


def process_batch(
    doc_name: str,
    pdf_paths: list,
    output_dir: str = "output",
    caption_prompt: str = DEFAULT_PROMPT,
    use_cache: bool = True,
):
    doc_output_dir = os.path.join(output_dir, doc_name)
    os.makedirs(doc_output_dir, exist_ok=True)

    logger.info(f"[1/7] Running OCR on {len(pdf_paths)} file(s) and stitching")
    combined_pages, pdf_map = stitch_results(pdf_paths, output_dir, use_cache=use_cache)
    results = {doc_name: combined_pages}

    results_path = os.path.join(doc_output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    logger.info("[2/7] Building hierarchy (sections, page-break AND file-break merges)")
    tree = build_document_tree(results, doc_name)

    logger.info("[3/7] Describing figures/pictures/diagrams")
    caption_figures(tree, pdf_map, caption_prompt, use_cache=use_cache)

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
    logger.info(f"Found {len(references['references'])} reference(s)")

    ast_path = os.path.join(doc_output_dir, "ast.json")
    with open(ast_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Wrote {ast_path}")
    return ast_path


def main():
    parser = argparse.ArgumentParser(description="Stitch multiple PDF excerpts into one AST")
    parser.add_argument("doc_name", help="Name for the combined document (output/<doc_name>/)")
    parser.add_argument("pdf_paths", nargs="+", help="PDF files, in the order they should be stitched")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--caption-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    process_batch(args.doc_name, args.pdf_paths, args.output_dir, args.caption_prompt, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
