import re
from typing import List, Tuple

import numpy
import torch

from surya.common.polygon import PolygonBox
from surya.recognition.schema import TextLine, TextWord, TextChar

MATH_SYMBOLS = ["+", "-", "*", "=", "^", "_", "\\", "{", "}"]


def unwrap_math(text: str) -> str:
    if len(text) > 50:
        return text

    # Detected as math, but does not contain LaTeX commands
    if (
        re.match(r'^\s*<math(?:\s+display="inline")?.*?</math>\s*$', text, re.DOTALL)
        and text.count("<math") == 1
        and not any([symb in text for symb in MATH_SYMBOLS])
    ):
        # Remove math tags
        text = re.sub(r"<math.*?>", "", text)
        text = re.sub(r"</math>", "", text)

    return text


MATH_BLOCK = re.compile(r"(<math\b[^>]*>)(.*?)</math>", flags=re.I | re.S)
STRIP_TAGS = re.compile(r"</?(?:br|u|del|mark|i|b|sup|sub)\b[^>]*>", flags=re.I | re.S)


def clean_math_tags(html: str) -> str:
    # strip unwanted tags inside every well‑formed <math>…</math>
    def _inner(m):
        inner = STRIP_TAGS.sub("", m.group(2))
        return f"{m.group(1)}{inner}</math>" if inner.strip() else ""

    cleaned = MATH_BLOCK.sub(_inner, html)

    # drop only orphan *closing* </math> tags
    depth = 0
    parts = []
    for token in re.split(r"(</?math[^>]*>)", cleaned, flags=re.I):
        if token.lower().startswith("<math"):
            depth += 1
            parts.append(token)
        elif token.lower() == "</math>":
            if depth:  # keep it only if it matches an open
                depth -= 1
                parts.append(token)
            # else: skip orphan closing tag
        else:
            parts.append(token)
    return "".join(parts)


def sort_text_lines(lines: List[TextLine] | List[dict], tolerance=1.25):
    # Sorts in reading order.  Not 100% accurate, this should only
    # be used as a starting point for more advanced sorting.
    vertical_groups = {}
    for line in lines:
        group_key = (
            round(
                line.bbox[1]
                if isinstance(line, TextLine)
                else line["bbox"][1] / tolerance
            )
            * tolerance
        )
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(line)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_lines = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(
            group, key=lambda x: x.bbox[0] if isinstance(x, TextLine) else x["bbox"][0]
        )
        sorted_lines.extend(sorted_group)

    return sorted_lines


def clean_close_polygons(bboxes: List[List[List[int]]], thresh: float = 0.1):
    if len(bboxes) < 2:
        return bboxes

    new_bboxes = [bboxes[0]]
    for i in range(1, len(bboxes)):
        close = True
        prev_bbox = bboxes[i - 1]
        bbox = bboxes[i]
        for j in range(4):
            if (
                abs(bbox[j][0] - prev_bbox[j][0]) > thresh
                or abs(bbox[j][1] - prev_bbox[j][1]) > thresh
            ):
                close = False
                break

        if not close:
            new_bboxes.append(bboxes[i])

    return new_bboxes


def words_from_chars(chars: List[TextChar], line_box: PolygonBox):
    words = []
    word = None
    for i, char in enumerate(chars):
        if not char.bbox_valid:
            if word:
                words.append(word)
                word = None
            continue

        if not word:
            word = TextWord(**char.model_dump())

            # Fit bounds to line if first word
            if i == 0:
                word.merge_left(line_box)

        elif not char.text.strip():
            if word:
                words.append(word)
            word = None
        else:
            # Merge bboxes
            word.merge(char)
            word.text = word.text + char.text

            if i == len(chars) - 1:
                word.merge_right(line_box)
    if word:
        words.append(word)

    return words