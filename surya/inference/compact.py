"""Surya 2.5 output format.

Instead of `<div data-bbox="71 33 143 45" data-label="Page-Header">`, a top-level block
header is one label token plus four coordinate tokens (0-1000): `<|Page-Header|><|71|><|33|><|143|><|45|>`.
A block has no `</div>`: the next header, or the end of the output, closes it. Other differences:

  - Tags are written in one canonical spelling (attribute order, quotes, boolean attributes).
  - `</td>`, `</th>` and `</tr>` are left out; decode() puts them back.
  - A `data-bbox` on any other tag (a row box, an image nested in a block) becomes
    `<|box|>` plus four coordinate tokens just before that tag.
  - Newlines next to block-level tags are dropped. Text inside <pre>, <code> and
    <math> is never touched.

decode() turns this back into the `<div data-label data-bbox>` HTML that parsers.py reads. Tags
the format does not know stay as literal HTML, so it is lossless without an exhaustive vocabulary.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Every label that can head a top-level block (LAYOUT_LABELS, LAYOUT_LABELS_ARXIV).
BLOCK_LABELS = [
    "Caption",
    "Footnote",
    "Equation-Block",
    "List-Group",
    "Page-Header",
    "Page-Footer",
    "Image",
    "Section-Header",
    "Table",
    "Text",
    "Complex-Block",
    "Code-Block",
    "Form",
    "Table-Of-Contents",
    "Figure",
    "Chemical-Block",
    "Diagram",
    "Bibliography",
    "Blank-Page",
    "Picture",
]
MAX_COORD = 1000

# Opening tags are canonicalized with attributes in this order; others follow sorted.
_ATTR_ORDER = [
    "type",
    "checked",
    "value",
    "colspan",
    "rowspan",
    "data-cs",
    "data-rs",
    "data-kind",
    "display",
    "class",
    "align",
    "style",
    "border",
    "href",
    "alt",
    "text",
]
_BOOLEAN_ATTRS = {"checked"}
_VOID = {"br", "hr", "img", "input"}
# Tags next to which a newline is formatting, not content.
_BLOCK_TAGS = {
    "div", "p", "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "br", "caption", "comment",
}  # fmt: skip
_OPAQUE = {"pre", "code", "math"}  # text inside is kept byte for byte
_OMIT_END = {"td", "th", "tr"}

_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)((?:\s+[^<>]*?)?)\s*(/?)>")
_ATTR_RE = re.compile(
    r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?"""
)
_LABELS_RE = "|".join(map(re.escape, BLOCK_LABELS))
_HEADER_RE = re.compile(r"<\|(%s)\|>" % _LABELS_RE + r"<\|(\d{1,4})\|>" * 4)
# A data-bbox on a tag other than a block header: <|box|> + four coordinates, then
# the tag without it. decode() puts the attribute back.
BOX = "<|box|>"
_BOX_VALUE_RE = re.compile(r"(0|[1-9]\d{0,3})( (0|[1-9]\d{0,3})){3}")


def _coords(prefix: str) -> str:
    return "".join(rf"<\|(?P<{prefix}{k}>\d{{1,4}})\|>" for k in range(4))


_DECODE_RE = re.compile(
    rf"(?P<hdr><\|(?P<label>{_LABELS_RE})\|>{_coords('h')})"
    rf"|(?P<box><\|box\|>{_coords('b')})"
    r"|(?P<tag><(?P<tclose>/?)(?P<tname>[a-zA-Z][a-zA-Z0-9-]*)"
    r"(?P<tattrs>(?:\s+[^<>]*?)?)\s*(?P<tself>/?)>)"
)


_ASCII_WS = " \t\n\r\f"


def _is_ws(s: str) -> bool:
    """ASCII whitespace only. A no-break space (U+00A0) is content: `<td>\xa0</td>`."""
    return s.strip(_ASCII_WS) == ""


def _parse_attrs(s: str) -> List[Tuple[str, Optional[str]]]:
    out = []
    for m in _ATTR_RE.finditer(s):
        name = m.group(1).lower()
        val = next((g for g in m.group(2, 3, 4) if g is not None), None)
        out.append((name, val))
    return out


def _attr_key(name: str):
    return (
        (_ATTR_ORDER.index(name), "")
        if name in _ATTR_ORDER
        else (len(_ATTR_ORDER), name)
    )


def canonical_tag(close: str, name: str, attrs: str, selfclose: str) -> str:
    """One spelling per tag: lowercase name, fixed attribute order, double quotes,
    `checked` without a value, void elements as `<x/>`."""
    name = name.lower()
    if close:
        return f"</{name}>"
    parts = []
    for k, v in sorted(_parse_attrs(attrs), key=lambda kv: _attr_key(kv[0])):
        if k in _BOOLEAN_ATTRS:
            parts.append(k)
        elif v is None:
            parts.append(f'{k}=""')
        else:
            # Values are re-quoted with ", so a literal " in a value must be escaped.
            parts.append(f'{k}="{v.replace(chr(34), "&quot;")}"')
    inner = name + ("" if not parts else " " + " ".join(parts))
    return f"<{inner}/>" if (name in _VOID or selfclose) else f"<{inner}>"


def _split(html: str):
    """Yield ('tag', raw, close, name, attrs, selfclose) or ('text', raw)."""
    pos = 0
    for m in _TAG_RE.finditer(html):
        if m.start() > pos:
            yield ("text", html[pos : m.start()])
        yield (
            "tag",
            m.group(0),
            m.group(1),
            m.group(2).lower(),
            m.group(3),
            m.group(4),
        )
        pos = m.end()
    if pos < len(html):
        yield ("text", html[pos:])


def _block_header(attrs: str) -> Optional[Tuple[str, List[int]]]:
    """(label, bbox) when a div carries exactly data-bbox and data-label."""
    a = dict(_parse_attrs(attrs))
    if set(a) != {"data-bbox", "data-label"} or a["data-label"] not in BLOCK_LABELS:
        return None
    try:
        box = [int(x) for x in (a["data-bbox"] or "").split()]
    except ValueError:
        return None
    if len(box) != 4 or not all(0 <= v <= MAX_COORD for v in box):
        return None
    return a["data-label"], box


def _split_box(attrs: str) -> Tuple[Optional[str], str]:
    """(box, attrs without data-bbox) when the tag has a canonical data-bbox, else
    (None, attrs). Only a canonical value qualifies, so decode() writes back the same
    string. The remaining attributes are rebuilt, so the tag stays canonical."""
    pairs = _parse_attrs(attrs)
    box = dict(pairs).get("data-bbox")
    if box is None or not _BOX_VALUE_RE.fullmatch(box):
        return None, attrs
    if any(int(v) > MAX_COORD for v in box.split()):
        return None, attrs
    rest = []
    for k, v in pairs:
        if k == "data-bbox":
            continue
        if v is None:
            rest.append(k)
        else:
            q = "'" if '"' in v and "'" not in v else '"'
            rest.append(f"{k}={q}{v}{q}")
    return box, (" " + " ".join(rest)) if rest else ""


def _next_is_header_or_end(items, j: int) -> bool:
    """A block's </div> may be left out only when a header or the end follows."""
    while j < len(items) and items[j][0] == "text" and _is_ws(items[j][1]):
        j += 1
    if j == len(items):
        return True
    it = items[j]
    return (
        it[0] == "tag"
        and not it[2]
        and it[3] == "div"
        and _block_header(it[4]) is not None
    )


# What may follow a left-out end tag: exactly the places where decode() restores it.
_IMPLIES_END = {
    "td": {"td", "th", "tr", "/tr", "/table", "/thead", "/tbody", "thead", "tbody"},
    "th": {"td", "th", "tr", "/tr", "/table", "/thead", "/tbody", "thead", "tbody"},
    "tr": {"tr", "/table", "/thead", "/tbody", "thead", "tbody"},
}


def _end_is_implied(items, j: int, name: str) -> bool:
    while j < len(items) and items[j][0] == "text" and _is_ws(items[j][1]):
        j += 1
    if j == len(items):
        return True
    it = items[j]
    if it[0] != "tag":
        return False
    if it[3] == "div":
        return (it[2] == "/") or _block_header(it[4]) is not None
    return (it[2] + it[3]) in _IMPLIES_END[name]


def _walk(html: str, compact: bool) -> str:
    """Shared by canonicalize (compact=False) and encode (compact=True): the two
    outputs differ only in the block headers and the omitted end tags."""
    items = list(_split(html))
    out: List[str] = []
    stack: List[str] = []  # open element names
    block_depth: Optional[int] = None  # stack depth of the open top-level block div
    opaque = 0

    def neighbor_is_block(i: int, step: int) -> bool:
        j = i + step
        return (
            0 <= j < len(items) and items[j][0] == "tag" and items[j][3] in _BLOCK_TAGS
        )

    after_header = False  # leading whitespace of a block's content is formatting
    for i, it in enumerate(items):
        if it[0] == "text":
            txt = it[1]
            if after_header and not opaque:
                txt = txt.lstrip(_ASCII_WS)
                if not txt:
                    continue
            after_header = False
            if not opaque and _is_ws(txt):
                prev_b, next_b = neighbor_is_block(i, -1), neighbor_is_block(i, 1)
                if (
                    (prev_b and next_b)
                    or not stack
                    or ("\n" in txt and (prev_b or next_b))
                ):
                    continue
            out.append(txt)
            continue
        _, raw, close, name, attrs, selfclose = it
        after_header = False
        if opaque and name not in _OPAQUE:
            out.append(raw)
            continue
        if not close:
            if name == "div" and not stack:
                hb = _block_header(attrs)
                if hb is not None:
                    label, box = hb
                    if out and not compact:
                        out.append("\n")  # canonical: one newline between blocks
                    if compact:
                        out.append(f"<|{label}|>" + "".join(f"<|{v}|>" for v in box))
                    else:
                        out.append(
                            f'<div data-bbox="{" ".join(map(str, box))}" data-label="{label}">'
                        )
                    stack.append("div")
                    block_depth = len(stack)
                    after_header = True
                    continue
            if compact:
                box, attrs = _split_box(attrs)
                if box is not None:
                    out.append(BOX + "".join(f"<|{v}|>" for v in box.split()))
            if name in _OPAQUE:
                opaque += 1
            out.append(canonical_tag("", name, attrs, selfclose))
            if name not in _VOID and not selfclose:
                stack.append(name)
            continue
        # closing tag
        if name in _OPAQUE and opaque:
            opaque -= 1
        if name in stack:
            while stack and stack[-1] != name:
                stack.pop()
            stack.pop()
        if name == "div" and block_depth is not None and len(stack) == block_depth - 1:
            block_depth = None
            # A newline before a block's close is formatting: the compact form has no
            # close to keep it against.
            if out and not out[-1].startswith("<"):
                out[-1] = out[-1].rstrip("\n")
            if compact and _next_is_header_or_end(items, i + 1):
                continue
        elif compact and name in _OMIT_END and _end_is_implied(items, i + 1, name):
            continue
        out.append(f"</{name}>")
    return "".join(out).strip("\n")


def canonicalize(html: str) -> str:
    """HTML in the canonical spelling the compact format round-trips to."""
    return _walk(html, compact=False)


def encode(html: str) -> str:
    """HTML -> compact output."""
    return _walk(html, compact=True)


def decode(text: str) -> str:
    """Compact output -> canonical HTML (the format every downstream parser reads).

    Tolerant of model output: explicit </td></th></tr></div> are accepted, unknown
    markup passes through, and a truncated output closes what it opened."""
    out: List[str] = []
    stack: List[str] = []
    in_block = False

    def close_cells(upto: Tuple[str, ...]):
        # Close omitted td/th/tr up to (not past) the nearest element in `upto`.
        while stack and stack[-1] in _OMIT_END and stack[-1] not in upto:
            out.append(f"</{stack.pop()}>")

    def close_block():
        nonlocal in_block
        if in_block:
            close_cells(())
            while stack and stack[-1] in _OMIT_END:
                out.append(f"</{stack.pop()}>")
            if stack and stack[-1] == "div":
                stack.pop()
            out.append("</div>")
            in_block = False

    pos = 0
    pending_box: Optional[str] = None  # a <|box|> waiting for its tag
    for m in _DECODE_RE.finditer(text):
        seg = text[pos : m.start()]
        if seg:
            out.append(seg)
        pos = m.end()
        g = m.group(0)
        if m.group("box"):
            pending_box = " ".join(str(int(m.group(f"b{k}"))) for k in range(4))
            continue
        if m.group("hdr"):
            pending_box = None
            label = m.group("label")
            box = [m.group(f"h{k}") for k in range(4)]
            close_block()
            if out and not out[-1].endswith("\n"):
                out.append("\n")
            out.append(
                f'<div data-bbox="{" ".join(str(int(b)) for b in box)}" data-label="{label}">'
            )
            stack.clear()
            stack.append("div")
            in_block = True
            continue
        close, name, selfclose = (
            m.group("tclose"),
            m.group("tname").lower(),
            m.group("tself"),
        )
        if pending_box is not None and not close:
            g = canonical_tag(
                "", name, f'{m.group("tattrs")} data-bbox="{pending_box}"', selfclose
            )
        pending_box = None
        if not close:
            if name in ("td", "th"):
                if stack and stack[-1] in ("td", "th"):
                    out.append(f"</{stack.pop()}>")
            elif name == "tr":
                close_cells(("table", "thead", "tbody"))
            elif name in ("thead", "tbody"):
                close_cells(("table",))
            out.append(g)
            if name not in _VOID and not selfclose:
                stack.append(name)
            continue
        if name in ("table", "thead", "tbody"):
            close_cells(("table", "thead", "tbody") if name != "table" else ("table",))
        if name == "div" and in_block and stack == ["div"]:
            close_block()
            continue
        if name in stack:
            while stack and stack[-1] != name:
                top = stack.pop()
                if top in _OMIT_END:
                    out.append(f"</{top}>")
            stack.pop()
        out.append(g)
    if pos < len(text):
        out.append(text[pos:])
    close_block()
    return "".join(out)
