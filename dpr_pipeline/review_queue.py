"""Flag everything in an AST tree that's worth a human double-checking,
instead of requiring a full manual re-read of the document.

Usage:
    uv run python -m dpr_pipeline.review_queue ast.json [--threshold 0.98] [--output review.json]

Checks (each backed by data already present in the tree, nothing new to run):
  - low_confidence: nodes whose OCR confidence is below --threshold. Note
    surya's full-page OCR produces one confidence score per page, broadcast
    to every block on it — so this generally flags whole pages, not
    individual blocks, which is still exactly the review granularity that
    matters (which pages need a second look).
  - caption_failures: figure/picture/diagram nodes where InternVL3 captioning
    errored (see caption_figures.py) — caption_error is set instead of a
    real description.
  - uncaptioned_figures: figure-type nodes that still carry the raw
    "[Label region — no OCR text]" placeholder — caption_figures.py was
    never run, or somehow left one behind.
  - numbering_gaps: siblings with outline numbering (e.g. "1.2.1", "1.2.2")
    where a number in the sequence is missing (e.g. "1.2.3" absent between
    "1.2.2" and "1.2.4") — usually means a heading's numbering was
    misread/missed, or (less often) real content is missing from the source.
"""

import argparse
import json
import re

TAG_RE = re.compile(r"<[^>]+>")
PLACEHOLDER_RE = re.compile(r"^\[.+ region — no OCR text\]$")


def find_low_confidence(node, threshold, path=()):
    conf = node.get("confidence")
    if conf is not None and conf < threshold:
        yield {
            "path": " > ".join(path),
            "page": node.get("page"),
            "type": node.get("type"),
            "label": node.get("label") or f"section(h{node.get('level')})",
            "confidence": conf,
            "preview": _preview(node),
        }
    label = node.get("title") or node.get("label")
    for c in node.get("children", []):
        yield from find_low_confidence(c, threshold, path + (label,) if label else path)


def find_caption_failures(node, path=()):
    if node.get("caption_error"):
        yield {
            "path": " > ".join(path),
            "page": node.get("page"),
            "label": node.get("label"),
            "error": node["caption_error"],
        }
    label = node.get("title") or node.get("label")
    for c in node.get("children", []):
        yield from find_caption_failures(c, path + (label,) if label else path)


def find_uncaptioned_figures(node, path=()):
    if node.get("type") == "block" and not node.get("caption_source") and not node.get("caption_error"):
        text = TAG_RE.sub("", node.get("html", "")).strip()
        if PLACEHOLDER_RE.match(text):
            yield {"path": " > ".join(path), "page": node.get("page"), "label": node.get("label")}
    label = node.get("title") or node.get("label")
    for c in node.get("children", []):
        yield from find_uncaptioned_figures(c, path + (label,) if label else path)


def find_numbering_gaps(node, path=()):
    numbered_children = [c for c in node.get("children", []) if c.get("type") == "section" and c.get("numbering")]
    by_depth = {}
    for c in numbered_children:
        by_depth.setdefault(len(c["numbering"]), []).append(c)

    for group in by_depth.values():
        if len(group) < 2:
            continue
        lasts = sorted(c["numbering"][-1] for c in group)
        missing = [n for n in range(lasts[0], lasts[-1] + 1) if n not in lasts]
        if missing:
            prefix = group[0]["numbering"][:-1]
            yield {
                "path": " > ".join(path),
                "prefix": ".".join(str(p) for p in prefix) or "(top level)",
                "present": lasts,
                "missing": missing,
            }

    label = node.get("title") or node.get("label")
    for c in node.get("children", []):
        yield from find_numbering_gaps(c, path + (label,) if label else path)


def _preview(node) -> str:
    if node.get("type") == "section":
        return node.get("title", "")
    return TAG_RE.sub(" ", node.get("html", "")).strip()[:80]


def build_review_queue(tree: dict, threshold: float = 0.98) -> dict:
    return {
        "low_confidence": list(find_low_confidence(tree, threshold)),
        "caption_failures": list(find_caption_failures(tree)),
        "uncaptioned_figures": list(find_uncaptioned_figures(tree)),
        "numbering_gaps": list(find_numbering_gaps(tree)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a review queue for an AST tree")
    parser.add_argument("tree_json")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--output", help="Write the report as JSON here (default: print a summary)")
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    report = build_review_queue(tree, args.threshold)

    total = sum(len(v) for v in report.values())
    print(f"Review queue: {total} item(s) flagged")
    for category, items in report.items():
        print(f"\n{category} ({len(items)}):")
        for item in items[:10]:
            print(" ", item)
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote full report to {args.output}")
