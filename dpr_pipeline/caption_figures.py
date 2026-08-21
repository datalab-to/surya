"""Fill in real descriptions for Figure/Picture/Diagram/... leaves in an AST
tree (built by build_ast.py) using surya.figure_caption.analyze_image.

Usage:
    uv run python -m dpr_pipeline.caption_figures TREE_JSON PDF_PATH [--prompt "..."] [--output OUT_JSON]

Crops each figure-type leaf's bbox out of the PDF page it belongs to (at the
same DPI the OCR bboxes were computed at), runs it through InternVL3, and
replaces the leaf's placeholder html with the model's response.

`pdf_path` is normally a single path, but process_batch.py's stitched trees
span multiple source PDFs with renumbered global pages — pass a
{source_file: pdf_path} dict in that case, and each node's own
source_file/source_page_local (set by build_ast.py only when present)
picks the right file and physical page instead of the single path/page.
"""

import argparse
import io
import json
import os
import tempfile

from surya.figure_caption import analyze_image
from surya.input.load import load_from_file
from surya.settings import settings

from . import cache

FIGURE_LABELS = {"Figure", "Picture", "Diagram", "Image", "Form", "ChemicalBlock"}

DEFAULT_PROMPT = (
    "Describe this image in detail. If it is a diagram, drawing, chart, or "
    "photograph, be specific and factual about what it depicts."
)

CAPTION_CACHE_NAMESPACE = "figure_caption"


def caption_figures(tree: dict, pdf_path, prompt: str = DEFAULT_PROMPT, use_cache: bool = True) -> dict:
    is_multi_source = isinstance(pdf_path, dict)
    images_by_path = {}

    def get_images(path):
        if path not in images_by_path:
            images_by_path[path], _ = load_from_file(path, dpi=settings.IMAGE_DPI_HIGHRES)
        return images_by_path[path]

    def walk(node):
        if node.get("type") == "block" and node.get("label") in FIGURE_LABELS:
            if is_multi_source:
                images = get_images(pdf_path[node["source_file"]])
                page_idx = node["source_page_local"] - 1
            else:
                images = get_images(pdf_path)
                page_idx = node["page"] - 1
            bbox = node["bbox"]
            crop = images[page_idx].crop(bbox)

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_key = cache.hash_bytes(
                buf.getvalue(),
                prompt.encode(),
                settings.FIGURE_CAPTION_MODEL_CHECKPOINT.encode(),
            )

            result = cache.get(CAPTION_CACHE_NAMESPACE, crop_key) if use_cache else None
            if result is None:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                try:
                    crop.save(tmp_path)
                    result = analyze_image(tmp_path, prompt)
                finally:
                    os.unlink(tmp_path)
                if use_cache:
                    cache.set(CAPTION_CACHE_NAMESPACE, crop_key, result)

            if result["success"]:
                node["html"] = f"<p>{result['response']}</p>"
                node["caption_source"] = "InternVL3-1B-Instruct"
            else:
                node["caption_error"] = result["error"]

        for child in node.get("children", []):
            walk(child)

    walk(tree)
    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption Figure/Picture/Diagram leaves in an AST tree")
    parser.add_argument("tree_json")
    parser.add_argument("pdf_path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", help="Path to write the updated tree (default: overwrite tree_json)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-captioning instead of using cached results")
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    caption_figures(tree, args.pdf_path, args.prompt, use_cache=not args.no_cache)

    output_path = args.output or args.tree_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    print(f"Wrote captioned tree to {output_path}")
