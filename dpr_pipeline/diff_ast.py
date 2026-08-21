"""Structurally diff two AST trees — e.g. a DPR and its "Revised DPR" — to
find which sections were added, removed, or had their content change,
without needing to manually re-read both documents side by side.

Usage:
    uv run python -m dpr_pipeline.diff_ast old_ast.json new_ast.json [--output diff.json]

Sections are matched across the two trees by outline numbering (e.g. "1.2.1"
— stable across revisions even if content around it changes) and, for
unnumbered headings, by title text. Matched sections are then compared by
their own direct text content (not their subsections', which are matched
and diffed independently) using difflib, so a changed CBR value or added
paragraph shows as a real text diff rather than just "section changed".
"""

import argparse
import difflib
import json
import re

TAG_RE = re.compile(r"<[^>]+>")


def node_identity(node):
    if node.get("numbering"):
        return ("numbering", tuple(node["numbering"]))
    return ("title", node.get("title", "").strip().lower())


def own_text(node) -> str:
    """This section's own text, from itself and its block descendants —
    stopping at any nested subsection, whose content is diffed separately."""
    parts = []

    def walk(n, is_root):
        if n.get("type") == "section" and not is_root:
            return
        if n.get("type") == "block":
            parts.append(TAG_RE.sub(" ", n.get("html", "")))
        for child in n.get("children", []):
            walk(child, False)

    walk(node, True)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def section_children(node):
    return [c for c in node.get("children", []) if c.get("type") == "section"]


def diff_sections(old_node, new_node, path=()):
    added, removed, changed, unchanged = [], [], [], []

    old_children = section_children(old_node)
    new_children = section_children(new_node)
    old_by_id = {node_identity(c): c for c in old_children}
    new_by_id = {node_identity(c): c for c in new_children}

    for identity, new_child in new_by_id.items():
        if identity not in old_by_id:
            added.append({"path": " > ".join(path), "title": new_child.get("title"), "page": new_child.get("page")})

    for identity, old_child in old_by_id.items():
        if identity not in new_by_id:
            removed.append({"path": " > ".join(path), "title": old_child.get("title"), "page": old_child.get("page")})

    for identity, old_child in old_by_id.items():
        new_child = new_by_id.get(identity)
        if new_child is None:
            continue

        old_text, new_text = own_text(old_child), own_text(new_child)
        child_path = path + (new_child.get("title"),)
        if old_text != new_text:
            ratio = difflib.SequenceMatcher(None, old_text, new_text).ratio()
            diff_lines = list(difflib.unified_diff(
                old_text.split(". "), new_text.split(". "),
                lineterm="", n=0,
            ))
            changed.append({
                "path": " > ".join(child_path),
                "title": new_child.get("title"),
                "old_page": old_child.get("page"),
                "new_page": new_child.get("page"),
                "similarity": round(ratio, 3),
                "diff": diff_lines,
            })
        else:
            unchanged.append({"path": " > ".join(child_path), "title": new_child.get("title")})

        sub_result = diff_sections(old_child, new_child, child_path)
        added.extend(sub_result["added"])
        removed.extend(sub_result["removed"])
        changed.extend(sub_result["changed"])
        unchanged.extend(sub_result["unchanged"])

    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}


def diff_ast(old_tree: dict, new_tree: dict) -> dict:
    return diff_sections(old_tree, new_tree, path=(old_tree.get("title"),))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structurally diff two AST trees")
    parser.add_argument("old_json")
    parser.add_argument("new_json")
    parser.add_argument("--output", help="Write JSON here (default: print a summary)")
    args = parser.parse_args()

    with open(args.old_json, "r", encoding="utf-8") as f:
        old_tree = json.load(f)
    with open(args.new_json, "r", encoding="utf-8") as f:
        new_tree = json.load(f)

    result = diff_ast(old_tree, new_tree)

    print(f"{len(result['added'])} added, {len(result['removed'])} removed, "
          f"{len(result['changed'])} changed, {len(result['unchanged'])} unchanged")

    for section in result["added"]:
        print(f"  + [{section['page']}] {section['path']} > {section['title']}")
    for section in result["removed"]:
        print(f"  - [{section['page']}] {section['path']} > {section['title']}")
    for section in result["changed"]:
        print(f"  ~ {section['path']} (similarity {section['similarity']})")
        for line in section["diff"][:6]:
            print(f"      {line}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")
