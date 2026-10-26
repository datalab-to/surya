"""Surya 2.5 writes compact block headers; parse_full_page_html must read them (no GPU needed).

The sample is the first 1,500 characters of a real Surya 2.5 full-page answer (a public arXiv page), so it
also covers a truncated tail."""

from pathlib import Path

from surya.inference.parsers import parse_full_page_html

SAMPLE = (Path(__file__).parent / "assets" / "compact_full_page_sample.txt").read_text()


def test_compact_page_parses_to_blocks():
    blocks = parse_full_page_html(SAMPLE)
    assert blocks, "compact output produced no blocks"
    assert blocks[0].label == "Text"
    assert blocks[0].bbox == (120.0, 87.0, 881.0, 255.0)
    assert "<math>" in blocks[0].html and "resili" in blocks[0].html


def test_div_format_unchanged():
    div = '<div data-bbox="10 20 500 80" data-label="Text"><p>hello</p></div>'
    blocks = parse_full_page_html(div)
    assert (
        len(blocks) == 1
        and blocks[0].label == "Text"
        and blocks[0].bbox == (10.0, 20.0, 500.0, 80.0)
    )


def test_two_compact_blocks_and_terminal_text():
    text = "<|Section-Header|><|10|><|20|><|500|><|60|><h2>Intro</h2><|Text|><|10|><|70|><|900|><|300|><p>Body text"
    blocks = parse_full_page_html(text)
    assert [b.label for b in blocks] == ["Section-Header", "Text"]
    assert (
        blocks[1].bbox == (10.0, 70.0, 900.0, 300.0) and "Body text" in blocks[1].html
    )


def test_truncated_header_does_not_raise():
    parse_full_page_html(
        "<|Text|><|10|><|20|>"
    )  # header cut mid-way: may yield no blocks, must not raise


def test_table_cells_without_closing_tags():
    text = "<|Table|><|0|><|0|><|1000|><|500|><table><tr><td>a<td>b<tr><td>c<td>d"
    blocks = parse_full_page_html(text)
    assert (
        len(blocks) == 1
        and blocks[0].html.lower().count("<td") == 4
        and blocks[0].html.lower().count("</td>") == 4
    )


def test_clean_block_html_decodes_marker_only():
    from surya.inference.parsers import clean_block_html

    assert clean_block_html("<p>x</p>") == "<p>x</p>"
    out = clean_block_html(
        "<table><|box|><|0|><|0|><|100|><|50|><tr><td>a<td>b</table>"
    )
    assert "<|" not in out


def test_clean_block_html_restores_omitted_table_closers():
    from surya.inference.parsers import clean_block_html

    out = clean_block_html("<table><tr><td>a<td>b<tr><td>c<td>d</table>")
    assert out.lower().count("<td") == 4 and out.lower().count("</td>") == 4
    already_closed = "<table><tr><td>a</td></tr></table>"
    assert clean_block_html(already_closed) == already_closed


def test_truncated_output_is_flagged_on_last_block():
    ok = parse_full_page_html(
        "<|Text|><|10|><|20|><|900|><|300|><p>whole paragraph</p>"
    )
    assert ok and not ok[-1].truncated
    cut = parse_full_page_html(
        "<|Text|><|10|><|20|><|900|><|300|><p>one</p><|Table|><|0|><|400|><|1000|><|900|><table><tr><td>a<td>b<tr><td>c"
    )
    assert (
        cut and cut[-1].label == "Table" and cut[-1].truncated and not cut[0].truncated
    )
    mid_math = parse_full_page_html(
        "<|Text|><|10|><|20|><|900|><|300|><p>so <math>x^2 +"
    )
    assert mid_math and mid_math[-1].truncated


def test_decoded_table_has_expected_rows_and_cells():
    from bs4 import BeautifulSoup

    blocks = parse_full_page_html(
        "<|Table|><|0|><|0|><|1000|><|500|><table><tr><td>a<td>b<tr><td>c<td>d</table>"
    )
    soup = BeautifulSoup(blocks[0].html, "html.parser")
    rows = [[td.get_text() for td in tr.find_all("td")] for tr in soup.find_all("tr")]
    assert rows == [["a", "b"], ["c", "d"]]


MERGED = (
    '<table border="1"><thead><tr><th data-rs="2" data-kind="parent">Species<th data-cs="2" data-kind="parent">GenBank'
    '<th data-kind="ghost"><tr><th data-kind="ghost"><th>28S<th>cax1</thead><tbody><tr><td>a<td>1<td>2</tbody>'
)


def test_block_merged_cells_become_colspan_rowspan():
    from bs4 import BeautifulSoup

    from surya.inference.parsers import clean_block_html

    out = clean_block_html(MERGED)
    assert "data-kind" not in out and "ghost" not in out
    soup = BeautifulSoup(out, "html.parser")
    ths = soup.find_all("th")
    assert ths[0].get("rowspan") == "2" and ths[1].get("colspan") == "2"
    assert [len(tr.find_all(["td", "th"])) for tr in soup.find_all("tr")] == [2, 2, 3]


def test_full_page_merged_cells_become_colspan_rowspan():
    blocks = parse_full_page_html("<|Table|><|0|><|0|><|1000|><|500|>" + MERGED)
    assert blocks and "rowspan" in blocks[0].html and "data-kind" not in blocks[0].html


def test_truncation_heuristic_cases():
    from surya.inference.parsers import html_looks_truncated

    assert not html_looks_truncated("<p>whole paragraph</p><ul><li>a</li></ul>")
    assert not html_looks_truncated(
        "<p>uses a <divider> element name</p>"
    )  # tag-like text is not a <div>
    assert html_looks_truncated("<p>cut mid sentence")
    assert html_looks_truncated("<p>so <math>x^2 +")
    assert html_looks_truncated(
        "<p>x</p><|Table|><|0|><|10|>"
    )  # half-written compact header
    assert html_looks_truncated("<table><tr><td>a</td>")


def test_partial_next_header_flags_last_block():
    blocks = parse_full_page_html(
        "<|Text|><|10|><|20|><|900|><|300|><p>done</p><|Table|><|0|><|10|>"
    )
    assert blocks and blocks[-1].truncated
    blocks = parse_full_page_html(
        "<|Text|><|10|><|20|><|900|><|300|><p>cut mid sentence"
    )
    assert blocks and blocks[-1].truncated


def test_omitted_tr_closers_with_explicit_cell_closers():
    from surya.inference.parsers import clean_block_html

    out = clean_block_html("<table><tr><td>a</td><tr><td>b</td></table>")
    assert out.lower().count("<tr") == 2 and out.lower().count("</tr>") == 2
