"""Export every Table node in an AST tree to CSV + one multi-sheet Excel
workbook, properly expanding colspan/rowspan into a real grid (the header
rows in these documents are full of merged cells, e.g. "S. No." spanning two
header rows while "Layer Thickness (mm)" spans three columns above it).

Usage:
    uv run python -m dpr_pipeline.export_tables TREE_JSON [--output-dir DIR]

Writes DIR/tables.xlsx (one sheet per table) and DIR/table_NNN.csv for each.
Sheet/file names come from the Table's preceding Caption sibling when one
exists, falling back to "table_<page>_<n>".
"""

import argparse
import csv
import json
import os
import re

from bs4 import BeautifulSoup
from openpyxl import Workbook

TAG_RE = re.compile(r"<[^>]+>")


def html_table_to_grid(html: str) -> list:
    """<table> HTML (with colspan/rowspan) -> a rectangular list-of-lists."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    rows = table.find_all("tr")
    grid = []
    # pending[col] = (remaining_rowspan, value) for cells spanning down into future rows
    pending = {}

    for row_idx, row in enumerate(rows):
        while len(grid) <= row_idx:
            grid.append([])
        row_out = grid[row_idx]

        col = 0

        def place(col, value):
            while len(row_out) <= col:
                row_out.append("")
            row_out[col] = value

        # Fill in any cells carried down from a previous row's rowspan.
        carried_cols = sorted(c for c, (remaining, _) in pending.items() if remaining > row_idx)
        for c in carried_cols:
            remaining, value = pending[c]
            place(c, value)

        for cell in row.find_all(["td", "th"]):
            while col in pending and pending[col][0] > row_idx:
                col += 1
            text = cell.get_text(separator=" ", strip=True)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for i in range(colspan):
                place(col + i, text)
                if rowspan > 1:
                    pending[col + i] = (row_idx + rowspan, text)
            col += colspan

    width = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([""] * (width - len(r)))
    return grid


def sanitize_filename(name: str, fallback: str) -> str:
    name = name.strip() or fallback
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:80]


def find_tables_with_captions(root):
    """(table_node, caption_text_or_None) for every Table, in document order.

    The most recently seen Caption is tracked across the whole pre-order
    walk (not per sibling group) because build_ast.py's indentation nesting
    often puts a Caption *inside* the paragraph that introduces a table
    (e.g. "Plates 3 to 5... <Caption>Table 4-9: ...</Caption>" as children
    of that paragraph), one level removed from being the Table's direct
    sibling. Pre-order traversal still visits everything in the original
    reading order regardless of that nesting, so a single running "last
    caption" value is enough to associate each table with its title.
    """
    last_caption = [None]
    results = []

    def walk(node):
        label = node.get("label")
        if label == "Caption":
            last_caption[0] = TAG_RE.sub("", node["html"]).strip()
        elif label == "Table":
            results.append((node, last_caption[0]))
            last_caption[0] = None
        for child in node.get("children", []):
            walk(child)

    walk(root)
    return results


def export_tables(tree: dict, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)

    written = []
    used_names = set()
    for idx, (table_node, caption) in enumerate(find_tables_with_captions(tree), start=1):
        grid = html_table_to_grid(table_node["html"])
        if not grid:
            continue

        base_name = sanitize_filename(caption or "", f"table_p{table_node['page']}_{idx}")
        sheet_name = base_name[:31] or f"table_{idx}"
        unique_name, n = sheet_name, 1
        while unique_name in used_names:
            n += 1
            unique_name = f"{sheet_name[:28]}_{n}"
        used_names.add(unique_name)

        ws = wb.create_sheet(title=unique_name)
        for row in grid:
            ws.append(row)

        csv_path = os.path.join(output_dir, f"{sanitize_filename(base_name, f'table_{idx}')}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(grid)

        written.append({"caption": caption, "page": table_node["page"], "csv": csv_path, "rows": len(grid)})

    xlsx_path = os.path.join(output_dir, "tables.xlsx")
    if written:
        wb.save(xlsx_path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AST Table nodes to CSV + Excel")
    parser.add_argument("tree_json")
    parser.add_argument("--output-dir", default=None, help="Default: <tree_json's folder>/tables")
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.tree_json)), "tables")
    written = export_tables(tree, output_dir)

    print(f"Exported {len(written)} table(s) to {output_dir}")
    for w in written:
        print(f"  p{w['page']} ({w['rows']} rows): {w['caption'] or '(no caption)'}")
