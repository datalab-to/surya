"""Pull structured engineering parameters out of an AST tree: equation
results, keyword-tagged numeric mentions in prose, and table rows as
field:value records — every one traceable back to its page and section path.

Usage:
    uv run python -m dpr_pipeline.extract_parameters ast.json [--output parameters.json]

This is intentionally pattern-based, not a general NLP extractor: it looks
for "<name> = ... = <value> <unit>" in Equation nodes, a curated list of
engineering keywords (CBR, resilient modulus, thickness, strain, ...)
followed by a number+unit in Text nodes, and treats each table's header row
as field names for its data rows. Good enough to turn "read the whole
document to find the CBR value" into a lookup, without claiming to
understand arbitrary engineering prose.
"""

import argparse
import json
import re

from .build_ast import TAG_RE
from .export_tables import find_tables_with_captions, html_table_to_grid

KEYWORDS = [
    "resilient modulus", "CBR", "modulus", "thickness", "strain",
    "friction angle", "tilting angle", "angle", "reliability", "design life",
    "traffic", "msa", "axle load", "pressure", "load", "temperature",
    "density", "gradient", "slope", "length", "width", "depth", "diameter",
    "height", "span", "capacity", "speed", "radius", "elevation", "rainfall",
]
# Longest-first so "resilient modulus" matches before the bare "modulus" inside it.
KEYWORDS = sorted(set(KEYWORDS), key=len, reverse=True)

UNIT_RE = r"(mm|cm|km/h|kmph|km|m|kg|kN|MPa|GPa|Pa|%|°|deg(?:rees?)?|days?|months?|years?|hrs?|hours?|msa)"
# A number, optionally in "N × 10^-E" scientific form (as seen in these
# documents' equations after LaTeX cleanup), optionally followed by a unit.
NUMBER_RE = r"-?\d+(?:\.\d+)?"
SCI_RE = rf"({NUMBER_RE})(?:\s*×\s*10\^(-?\d+))?"
VALUE_UNIT_RE = re.compile(rf"{SCI_RE}\s*{UNIT_RE}?\b")

BRACE_CONTENT_RE = re.compile(r"\\text\{([^}]*)\}")
FRAC_RE = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")


def clean_math(html: str) -> str:
    text = TAG_RE.sub("", html)
    text = FRAC_RE.sub(r"(\1)/(\2)", text)
    text = BRACE_CONTENT_RE.sub(r"\1", text)
    text = text.replace("\\times", "×").replace("\\quad", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"[{}\\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_value(match) -> tuple:
    mantissa, exponent, unit = match.group(1), match.group(2), match.group(3)
    value = float(mantissa) * (10 ** int(exponent)) if exponent else float(mantissa)
    return value, unit


def extract_from_equation(node, path):
    expr = clean_math(node["html"])
    parts = [p.strip() for p in expr.split("=")]
    if len(parts) < 2:
        return None

    name = parts[0]
    if not re.match(r"^[A-Za-z]", name):
        # e.g. "5 × 10^6 = 2.21 × 10^-04 × [...]" is a substitution step
        # (checking a target value against a formula), not "5 × 10^6" being
        # defined as a named parameter — nothing useful to extract here.
        return None

    # Only the *last* "=" segment can be a resolved answer, and only if it's
    # cleanly just a number(+unit) with nothing else — an unevaluated
    # formula like "2.21×10⁻⁴ × [1/εt]^3.89 × [1/M_R]^0.854" also contains a
    # number, but grabbing it as if it were the parameter's value would be
    # wrong: it's a coefficient inside an equation with no numeric answer
    # yet, not a result.
    last = parts[-1].strip()
    match = VALUE_UNIT_RE.fullmatch(last)
    value, unit = resolve_value(match) if match else (None, None)

    return {
        "type": "equation",
        "path": " > ".join(path),
        "page": node["page"],
        "parameter": name,
        "value": value,
        "unit": unit,
        "expression": expr,
    }


def extract_from_text(node, path):
    text = re.sub(r"\s+", " ", TAG_RE.sub(" ", node["html"])).strip()
    lower = text.lower()
    results = []
    consumed = set()
    for kw in KEYWORDS:
        for kw_match in re.finditer(re.escape(kw.lower()), lower):
            if kw_match.start() in consumed:
                continue
            window = text[kw_match.start(): kw_match.start() + 120]
            value_match = VALUE_UNIT_RE.search(window, kw_match.end() - kw_match.start())
            if value_match:
                value, unit = resolve_value(value_match)
                results.append({
                    "type": "text_mention",
                    "path": " > ".join(path),
                    "page": node["page"],
                    "keyword": kw,
                    "value": value,
                    "unit": unit,
                    "snippet": window.strip(),
                })
                consumed.add(kw_match.start())
    return results


def looks_like_header_row(row) -> bool:
    """A second header row (from a merged colspan/rowspan header) reads as
    labels, not data — none of its cells parse as a bare number."""
    return all(not re.fullmatch(r"-?\d+\.?", cell.strip()) for cell in row if cell.strip())


def extract_from_tables(tree):
    results = []
    for table_node, caption in find_tables_with_captions(tree):
        grid = html_table_to_grid(table_node["html"])
        if len(grid) < 2:
            continue
        header = grid[0]
        data_start = 1
        if len(grid) > 2 and looks_like_header_row(grid[1]):
            header = [f"{a} - {b}" if a and a != b else (b or a) for a, b in zip(grid[0], grid[1])]
            data_start = 2
        for row in grid[data_start:]:
            record = {h: v for h, v in zip(header, row) if h}
            if any(v.strip() for v in record.values()):
                results.append({
                    "type": "table_row",
                    "table": caption,
                    "page": table_node["page"],
                    "record": record,
                })
    return results


def extract_parameters(tree: dict) -> dict:
    equations, text_mentions = [], []

    def walk(node, path=()):
        label = node.get("label")
        if label == "Equation":
            result = extract_from_equation(node, path)
            if result:
                equations.append(result)
        elif label in ("Text", "Caption"):
            text_mentions.extend(extract_from_text(node, path))

        node_label = node.get("title") or node.get("label")
        for child in node.get("children", []):
            walk(child, path + (node_label,) if node_label else path)

    walk(tree)

    return {
        "equations": equations,
        "text_mentions": text_mentions,
        "table_rows": extract_from_tables(tree),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured engineering parameters from an AST tree")
    parser.add_argument("tree_json")
    parser.add_argument("--output", help="Write JSON here (default: print a summary)")
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    params = extract_parameters(tree)

    for category, items in params.items():
        print(f"\n{category} ({len(items)}):")
        for item in items[:10]:
            print(" ", item)
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")
