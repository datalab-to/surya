"""Resolve "as shown in Table 4-10" / "see Figure 1-1" / "as per Section 4.9"
/ "refer to Chapter 1" style mentions in body text to the actual node they
name, turning the tree from a hierarchy into a lightly-linked graph.

Usage:
    uv run python -m dpr_pipeline.resolve_references ast.json [--output references.json]

A mention resolves only against *this* document's own captions/numbering —
text referencing a table/section/clause in a *different* document (e.g.
"Table 500-17 of ... Specification for Road and Bridge Works", a MORTH spec
table numbered in a scheme this DPR doesn't use, or "Clause 507" from that
same external spec) correctly comes back unresolved rather than a false
match. Nothing here searches by page number or physical proximity, only by
explicit caption/numbering text, so it can't confuse "the nearest Table"
with "the one actually named".
"""

import argparse
import json
import re

from .build_ast import TAG_RE

TARGET_LABELS = {"Table", "Figure", "Picture", "Diagram", "Image"}
CAPTION_NUM_RE = re.compile(r"\b(Table|Figure)\s*([\d]+(?:[-.][\d]+)*)", re.IGNORECASE)
CHAPTER_TITLE_RE = re.compile(r"^chapter\s+(\d+)\b", re.IGNORECASE)
# One combined pass per Text block: Table/Figure mentions look up a caption;
# Section/Chapter/Clause mentions look up a heading's own outline numbering
# (see collect_section_targets) — same regex, dispatched by which kind word
# was used, since "Clause 4.9" and "Section 4.9" both just mean "the heading
# numbered 4.9" as far as resolving is concerned.
MENTION_RE = re.compile(r"\b(Table|Figure|Fig\.|Section|Chapter|Clause)\s*([\d]+(?:[-.][\d]+)*)", re.IGNORECASE)

TABLE_FIGURE_KINDS = {"table", "figure", "fig."}


def normalize_key(kind: str, number: str) -> str:
    kind = "figure" if kind.lower().startswith("fig") else kind.lower()
    return f"{kind} {number.replace('.', '-')}"


def collect_targets(root: dict) -> dict:
    """key ("table 4-10") -> target node info, tracking the most recent
    Caption across the whole pre-order walk (same reasoning as
    export_tables.find_tables_with_captions: build_ast.py's indentation
    nesting can put a Caption inside the paragraph introducing a table/
    figure, one level removed from being its direct sibling). AI-generated
    figure descriptions (caption_source set) are also searched, since
    InternVL3 sometimes reads a figure's own printed label off the image
    even when surya never OCR'd a separate Caption block for it.
    """
    last_caption = [None]
    targets = {}

    def walk(node, path):
        label = node.get("label")
        if label == "Caption":
            last_caption[0] = TAG_RE.sub("", node["html"]).strip()
        elif label in TARGET_LABELS:
            caption_texts = [c for c in (last_caption[0], node.get("html") if node.get("caption_source") else None) if c]
            for caption in caption_texts:
                match = CAPTION_NUM_RE.search(TAG_RE.sub("", caption))
                if match:
                    key = normalize_key(match.group(1), match.group(2))
                    targets.setdefault(key, {
                        "label": label,
                        "caption": TAG_RE.sub("", last_caption[0]).strip() if last_caption[0] else None,
                        "page": node["page"],
                        "path": " > ".join(path),
                    })
            last_caption[0] = None

        node_label = node.get("title") or node.get("label")
        for child in node.get("children", []):
            walk(child, path + (node_label,) if node_label else path)

    walk(root, ())
    return targets


def collect_section_targets(root: dict) -> tuple:
    """(numbering -> node info, chapter number -> node info).

    Sections are keyed by their own outline numbering (e.g. "4.9", "4.9.1")
    regardless of whether body text calls it "Section 4.9" or "Clause 4.9"
    — both just mean "the heading numbered 4.9" for resolving purposes.
    Unnumbered "Chapter N" headings (numbering is None for these — they're
    titles, not outline-numbered subsections) are indexed separately by the
    chapter number in their title text, since that's the only way to match
    "refer to Chapter 1" mentions to them.
    """
    numbered, chapters = {}, {}

    def walk(node, path):
        if node.get("type") == "section":
            info = {"title": node.get("title"), "page": node.get("page"), "path": " > ".join(path)}
            if node.get("numbering"):
                numbered.setdefault(".".join(str(n) for n in node["numbering"]), info)
            chapter_match = CHAPTER_TITLE_RE.match((node.get("title") or "").strip())
            if chapter_match:
                chapters.setdefault(chapter_match.group(1), info)

        node_label = node.get("title") or node.get("label")
        for child in node.get("children", []):
            walk(child, path + (node_label,) if node_label else path)

    walk(root, ())
    return numbered, chapters


def find_references(root: dict, targets: dict, numbered_sections: dict, chapters: dict) -> list:
    results = []

    def walk(node, path):
        if node.get("label") == "Text":
            plain = TAG_RE.sub(" ", node["html"])
            for match in MENTION_RE.finditer(plain):
                kind, number = match.group(1), match.group(2)
                if kind.lower() in TABLE_FIGURE_KINDS:
                    resolved = targets.get(normalize_key(kind, number))
                else:
                    resolved = numbered_sections.get(number)
                    if resolved is None and kind.lower() == "chapter":
                        resolved = chapters.get(number)
                results.append({
                    "source_path": " > ".join(path),
                    "source_page": node.get("page"),
                    "mention": match.group(0),
                    "resolved": resolved,
                })

        node_label = node.get("title") or node.get("label")
        for child in node.get("children", []):
            walk(child, path + (node_label,) if node_label else path)

    walk(root, ())
    return results


def resolve_references(tree: dict) -> dict:
    targets = collect_targets(tree)
    numbered_sections, chapters = collect_section_targets(tree)
    references = find_references(tree, targets, numbered_sections, chapters)
    return {
        "targets": targets,
        "section_targets": numbered_sections,
        "chapter_targets": chapters,
        "references": references,
        "unresolved_count": sum(1 for r in references if r["resolved"] is None),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve Table/Figure/Section/Chapter mentions to their target nodes")
    parser.add_argument("tree_json")
    parser.add_argument("--output", help="Write JSON here (default: print a summary)")
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    result = resolve_references(tree)

    n_targets = len(result["targets"]) + len(result["section_targets"]) + len(result["chapter_targets"])
    print(f"{n_targets} target(s) ({len(result['targets'])} table/figure, {len(result['section_targets'])} numbered "
          f"section, {len(result['chapter_targets'])} chapter), {len(result['references'])} reference(s) found, "
          f"{result['unresolved_count']} unresolved")
    for ref in result["references"]:
        if ref["resolved"]:
            desc = ref["resolved"].get("caption") or ref["resolved"].get("title")
            status = f"-> {desc}"
        else:
            status = "(unresolved)"
        print(f"  [{ref['source_page']}] \"{ref['mention']}\" {status}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")
