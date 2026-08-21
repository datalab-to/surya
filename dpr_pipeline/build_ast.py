"""Build a section/subsection AST from surya OCR results.json.

surya's OCR output is a flat, per-page list of blocks with a reading_order
and (for SectionHeader blocks) an HTML heading tag (h1-h5) carrying the
visual heading depth. This walks that flat sequence across all pages in
two nested passes:

1. A heading-level stack (h1-h5) — same idea as building a DOM tree from
   a token stream — gives the section/subsection structure. The model's
   own h-tag is occasionally inconsistent across siblings that share a
   section number (e.g. "1.2.1" tagged h4 but its sibling "1.2.2" tagged
   h3, which would wrongly make 1.2.2 a sibling of "1.2" instead of its
   child) — see numbering_level_by_prefix below for the correction.
2. Within each section, an indentation stack: a block whose left edge
   (bbox x0) is indented past the block immediately before it, AND
   which starts further down the page (bbox y0) than that block,
   becomes that block's *child* rather than its sibling (e.g. a list
   introduced by "...as detailed below" is visually indented past that
   sentence and sits below it, so it nests under it). It pops back out
   once indentation returns to the section's base margin. This is a
   layout signal, not a text/phrase match, so it isn't tied to any
   particular wording or language.

   The y0 check exists specifically to rule out side-by-side content:
   two photo captions in a two-up image layout can have very different
   x0 (one is just physically to the right of the other) while sitting
   at the same height on the page — that's a row, not a hierarchy, and
   without the y0 check the righthand one would wrongly nest under the
   lefthand one.
3. Split-content merging: surya OCRs one page at a time, so any block
   whose content spills past the bottom of a page comes back as two
   separate blocks — one per page — instead of one. Whenever a block's
   label matches the immediately preceding leaf's label (nothing between
   them but skipped PageHeader/Footer) AND they're on different pages,
   a per-label check decides whether it's really a continuation, and if
   so its content is folded into the first block instead of creating a
   new sibling:
     - Table: matching repeated header cells, or (if the second fragment
       has no header at all — itself a signal, since a genuinely new
       table almost always starts with one) a matching column count.
     - ListGroup: always — two list fragments split by a page break are
       merged item-by-item into one list.
     - Text / Caption: only when the first fragment's plain text doesn't
       end in sentence-ending punctuation AND the second fragment starts
       with a lowercase letter — i.e. it reads as a sentence cut off
       mid-thought, not two genuinely separate paragraphs that happen to
       sit back-to-back across a page break.

Sections stay open across a page break automatically, with no special
handling needed: the heading stack above is built by walking every
block in every page in one flat pass, and a heading only closes when
another SectionHeader block pops it — the page number never enters
into that decision. So content that starts on the page after its
heading (e.g. a heading is the last line on a page, its paragraph is
the first thing on the next) nests correctly with no extra logic.

Caveat: the indentation/row heuristics assume blocks visually
below-and-indented are truly nested content, and blocks in the same
row are truly siblings — reasonable defaults for typical reports, but
a multi-column layout where reading order zig-zags in unusual ways can
still produce a case they don't cover. The ListGroup merge is the
riskiest of the continuation rules — it has no header/punctuation
signal to check, so two genuinely unrelated lists placed back-to-back
across a page break (no heading between them) would false-merge; this
is rarer than a real split list in practice, but worth spot-checking.
"""

import json
import re

HEADING_RE = re.compile(r"<h([1-5])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
# A leading numbered-outline prefix, e.g. "1.2.3" in "1.2.3. Chakabama Road"
# or "4.9.1" in "4.9.1. For CBR 5%". Requires whitespace + more text after,
# so things like "20% Increase" (digits not followed by a numbering-shaped
# separator) don't false-match.
NUMBERING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")

SKIP_LABELS = {"PageHeader", "PageFooter"}
# surya never runs OCR on these (SKIP_OCR_LABELS in surya/inference/prompts.py)
# since there's no text to extract — html is always "". They're still real
# structural elements of the document though, so keep them as placeholder
# leaves rather than dropping them the way a running header/footer is
# dropped. Anything else with skipped=True is a text-bearing block whose
# region turned out blank — that one has nothing worth keeping.
NO_OCR_LABELS = {"Figure", "Picture", "Diagram", "Image", "Form", "ChemicalBlock"}

# Minimum left-edge shift (as a fraction of page width) before a block is
# considered "indented" relative to the block before it, rather than just
# OCR/bbox jitter around the same margin.
INDENT_FRACTION = 0.008
MIN_INDENT_PX = 6


def heading_level_and_title(html: str):
    match = HEADING_RE.search(html)
    if match:
        return int(match.group(1)), TAG_RE.sub("", match.group(2)).strip()
    return None, TAG_RE.sub("", html).strip()


def parse_numbering(title: str):
    match = NUMBERING_RE.match(title)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


THEAD_RE = re.compile(r"<thead>(.*?)</thead>", re.IGNORECASE | re.DOTALL)
TABLE_OPEN_RE = re.compile(r"<table[^>]*>", re.IGNORECASE)
ROW_RE = re.compile(r"<tr[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[hd][^>]*>", re.IGNORECASE)
TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.IGNORECASE | re.DOTALL)


def table_header_cells(html: str):
    """Header cell texts if this table has a <thead>, else None."""
    thead = THEAD_RE.search(html)
    if not thead:
        return None
    cells = TH_RE.findall(thead.group(1))
    return tuple(TAG_RE.sub("", c).strip().lower() for c in cells)


def table_data_rows(html: str):
    """<tr> rows outside any <thead>, whether wrapped in <tbody> or bare."""
    return ROW_RE.findall(THEAD_RE.sub("", html))


def is_table_continuation(first_html: str, second_html: str):
    first_headers = table_header_cells(first_html)
    second_headers = table_header_cells(second_html)
    if first_headers is not None and second_headers is not None:
        # Repeated header row on the continuation page (e.g. long tables that
        # re-print column headers after a page break) — exact match required.
        return first_headers == second_headers

    if second_headers is None:
        # No header at all on the second fragment is itself a signal: a
        # genuinely new, standalone table almost always starts with one.
        # Corroborate with matching column count so an unrelated headerless
        # table (rare) doesn't false-merge.
        first_rows = table_data_rows(first_html)
        second_rows = table_data_rows(second_html)
        if not first_rows or not second_rows:
            return False
        return len(CELL_RE.findall(first_rows[-1])) == len(CELL_RE.findall(second_rows[0]))

    return False


def merge_tables(first_html: str, second_html: str):
    thead = THEAD_RE.search(first_html)
    head_part = first_html[: thead.end()] if thead else TABLE_OPEN_RE.match(first_html).group(0)
    merged_rows = "".join(table_data_rows(first_html)) + "".join(table_data_rows(second_html))
    return f"{head_part}<tbody>{merged_rows}</tbody></table>"


LIST_OPEN_RE = re.compile(r"<(ul|ol)([^>]*)>", re.IGNORECASE)
LI_RE = re.compile(r"<li[^>]*>.*?</li>", re.IGNORECASE | re.DOTALL)


def is_listgroup_continuation(first_html: str, second_html: str):
    return bool(LI_RE.search(first_html)) and bool(LI_RE.search(second_html))


def merge_listgroups(first_html: str, second_html: str):
    open_match = LIST_OPEN_RE.match(first_html)
    tag, attrs = (open_match.group(1), open_match.group(2)) if open_match else ("ul", "")
    merged_items = "".join(LI_RE.findall(first_html)) + "".join(LI_RE.findall(second_html))
    return f"<{tag}{attrs}>{merged_items}</{tag}>"


P_RE = re.compile(r"^\s*<p>(.*)</p>\s*$", re.IGNORECASE | re.DOTALL)
# Sentence-ending punctuation, allowing a trailing quote/paren (e.g. `done.")`).
SENTENCE_END_RE = re.compile(r"[.!?:;][\'\"’”)]*\s*$")


def plain_text(html: str) -> str:
    return TAG_RE.sub("", html).strip()


def is_text_continuation(first_html: str, second_html: str):
    first_text = plain_text(first_html)
    second_text = plain_text(second_html)
    if not first_text or not second_text:
        return False
    if SENTENCE_END_RE.search(first_text):
        return False
    return second_text[0].islower()


def merge_text_blocks(first_html: str, second_html: str):
    first_match = P_RE.match(first_html)
    second_match = P_RE.match(second_html)
    first_inner = first_match.group(1) if first_match else first_html
    second_inner = second_match.group(1) if second_match else second_html
    return f"<p>{first_inner.rstrip()} {second_inner.lstrip()}</p>"


def always_continuation(first_html: str, second_html: str):
    return True


def merge_raw_concat(first_html: str, second_html: str):
    # TableOfContents entries are the loosest-structured label surya emits —
    # the same logical TOC can come back as a <table> on one page and a
    # run of bare <p> lines on the next (observed in practice, not just
    # theoretical). Rather than force both fragments into one markup shape,
    # just place them one after another; a split TOC is effectively always
    # one continuous list, so no continuation check beyond "same label,
    # different page" is needed.
    return first_html + second_html


CONTINUATION_CHECKERS = {
    "Table": is_table_continuation,
    "ListGroup": is_listgroup_continuation,
    "Text": is_text_continuation,
    "Caption": is_text_continuation,
    "TableOfContents": always_continuation,
}
CONTINUATION_MERGERS = {
    "Table": merge_tables,
    "ListGroup": merge_listgroups,
    "Text": merge_text_blocks,
    "Caption": merge_text_blocks,
    "TableOfContents": merge_raw_concat,
}


def build_document_tree(results: dict, doc_name: str, fallback_heading_level: int = 1):
    root = {"type": "document", "title": doc_name, "children": []}
    # Each stack entry: (heading_level, node, indent_stack). indent_stack is a
    # list of (x0, node) tracking open indentation levels within this section.
    stack = [(0, root, [])]
    # parent numbering prefix (e.g. (1, 2) for "1.2.x" children) -> the level
    # its first-seen child was placed at, so later siblings under the same
    # prefix are forced to that level even if their own h-tag disagrees.
    numbering_level_by_prefix = {}
    # The most recently created content leaf (any label), kept only while
    # nothing but a SectionHeader-free, skip-label-only gap separates it from
    # the next block — reset to None whenever a heading intervenes. Used to
    # detect split content across a page break (see module docstring).
    last_leaf = [None]

    for page in results[doc_name]:
        page_num = page["page"]
        page_width = page.get("image_bbox", [0, 0, 0, 0])[2] or 1000
        indent_eps = max(MIN_INDENT_PX, INDENT_FRACTION * page_width)

        for block in sorted(page["blocks"], key=lambda b: b["reading_order"]):
            if block["label"] in SKIP_LABELS:
                continue
            if block.get("skipped") and block["label"] not in NO_OCR_LABELS:
                continue

            if block["label"] == "SectionHeader":
                level, title = heading_level_and_title(block["html"])
                if level is None:
                    level = fallback_heading_level

                numbering = parse_numbering(title)
                if numbering is not None:
                    parent_prefix = numbering[:-1]
                    if parent_prefix in numbering_level_by_prefix:
                        level = numbering_level_by_prefix[parent_prefix]
                    else:
                        numbering_level_by_prefix[parent_prefix] = level

                while stack[-1][0] >= level:
                    stack.pop()

                node = {
                    "type": "section",
                    "level": level,
                    "title": title,
                    "page": page_num,
                    "bbox": block["bbox"],
                    "confidence": block["confidence"],
                    "children": [],
                }
                if numbering is not None:
                    node["numbering"] = list(numbering)
                # Set only in multi-PDF batch mode (see process_batch.py),
                # where "page" is a renumbered global page — these carry the
                # original (file, local page number) for anything that needs
                # to go back to the actual PDF pixels.
                for key in ("source_file", "source_page_local"):
                    if key in block:
                        node[key] = block[key]
                stack[-1][1]["children"].append(node)
                stack.append((level, node, []))
                last_leaf[0] = None
            else:
                if last_leaf[0] is not None:
                    prev_label, prev_node = last_leaf[0]
                    checker = CONTINUATION_CHECKERS.get(block["label"])
                    if (
                        block["label"] == prev_label
                        and checker is not None
                        and prev_node["page"] < page_num
                        and checker(prev_node["html"], block["html"])
                    ):
                        prev_node["html"] = CONTINUATION_MERGERS[block["label"]](prev_node["html"], block["html"])
                        prev_node.setdefault("continued_on", []).append(
                            {"page": page_num, "bbox": block["bbox"]}
                        )
                        # The merged node's confidence is the worst of its
                        # fragments — a review queue should surface it if
                        # *either* half was uncertain.
                        prev_node["confidence"] = min(prev_node["confidence"], block["confidence"])
                        continue

                _, section_node, indent_stack = stack[-1]
                x0, y0 = block["bbox"][0], block["bbox"][1]

                while indent_stack:
                    top_x0, top_y0, top_node = indent_stack[-1]
                    not_indented = top_x0 >= x0 - indent_eps
                    same_row_or_above = y0 <= top_y0
                    if not_indented or same_row_or_above:
                        indent_stack.pop()
                    else:
                        break

                parent = indent_stack[-1][2] if indent_stack else section_node

                html = block["html"]
                if not html.strip():
                    html = f"<p><i>[{block['label']} region — no OCR text]</i></p>"

                leaf = {
                    "type": "block",
                    "label": block["label"],
                    "html": html,
                    "page": page_num,
                    "bbox": block["bbox"],
                    "reading_order": block["reading_order"],
                    "confidence": block["confidence"],
                    "children": [],
                }
                for key in ("source_file", "source_page_local"):
                    if key in block:
                        leaf[key] = block[key]
                parent["children"].append(leaf)
                last_leaf[0] = (block["label"], leaf)
                # A figure/picture/diagram never "introduces" what comes after
                # it the way a sentence or a table does, so it isn't eligible
                # to become a parent via indentation — just skip pushing it.
                if block["label"] not in NO_OCR_LABELS:
                    indent_stack.append((x0, y0, leaf))

    return root


def print_tree(node, indent=0):
    if node["type"] == "document":
        print(f"{'  ' * indent}# {node['title']}")
    elif node["type"] == "section":
        print(f"{'  ' * indent}{'#' * (node['level'] + 1)} [p{node['page']}] {node['title']}")
    else:
        preview = TAG_RE.sub(" ", node["html"]).strip()[:70]
        print(f"{'  ' * indent}- [p{node['page']}] ({node['label']}) {preview}")
    for child in node.get("children", []):
        print_tree(child, indent + 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a section/subsection AST from surya results.json")
    parser.add_argument("results_path")
    parser.add_argument("doc_name", nargs="?", default=None)
    parser.add_argument("--json-out", help="Write the JSON tree to this path instead of stdout")
    args = parser.parse_args()

    with open(args.results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    doc_name = args.doc_name or next(iter(results))
    tree = build_document_tree(results, doc_name)

    print_tree(tree)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON tree to {args.json_out}")
    else:
        print()
        print(json.dumps(tree, indent=2, ensure_ascii=False))
