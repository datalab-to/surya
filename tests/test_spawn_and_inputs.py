import pypdfium2
import pytest

from surya.inference.backends import spawn
from surya.inference.prompts import LAYOUT_JSON_SCHEMA
from surya.input.load import get_name_from_path, load_pdf
from surya.layout import LayoutPredictor


def test_get_name_from_path_preserves_dots():
    assert get_name_from_path("/some/dir/report.v1.2.final.pdf") == "report.v1.2.final"
    assert get_name_from_path("simple.png") == "simple"


def test_load_pdf_empty_page_range(tmp_path):
    pdf_path = tmp_path / "doc.v1.pdf"
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(200, 200)
    pdf.save(str(pdf_path))
    pdf.close()

    images, names = load_pdf(str(pdf_path), page_range=[])
    assert images == []
    assert names == []

    images_1, names_1 = load_pdf(str(pdf_path), page_range=[0])
    assert len(images_1) == 1
    assert names_1 == ["doc.v1"]


def test_layout_json_schema_uses_ascii_digits():
    pattern = LAYOUT_JSON_SCHEMA["items"]["properties"]["bbox"]["pattern"]
    assert pattern == r"^[0-9]{1,4} [0-9]{1,4} [0-9]{1,4} [0-9]{1,4}$"


def test_layout_predictor_rejects_non_pil():
    pred = LayoutPredictor()
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        pred(["some_file.pdf"])


def test_wait_for_health_aborts_early_when_process_dies(monkeypatch):
    monkeypatch.setattr(spawn, "probe_health", lambda url, timeout=1.0: False)
    calls = 0

    def is_alive():
        nonlocal calls
        calls += 1
        return False

    ok = spawn.wait_for_health(
        "http://127.0.0.1:9999", total_timeout=30.0, interval=0.01, is_alive=is_alive
    )
    assert ok is False
    assert calls == 1


def test_attach_or_spawn_supports_multi_model_server(monkeypatch):
    monkeypatch.setattr(spawn, "probe_health", lambda url, timeout=1.0: True)
    monkeypatch.setattr(
        spawn,
        "probe_model_ids",
        lambda url, timeout=5.0: ["other-model", "datalab-to/surya-ocr-2.5"],
    )

    srv = spawn.attach_or_spawn(
        backend="vllm",
        expected_model_name="datalab-to/surya-ocr-2.5",
        spawn_fn=lambda port: None,
        health_url_for=lambda port: f"http://127.0.0.1:{port}",
        openai_url_for=lambda port: f"http://127.0.0.1:{port}/v1",
        external_url="http://127.0.0.1:8000/v1",
    )
    assert srv.model_name == "datalab-to/surya-ocr-2.5"
    assert srv.spawned_by_us is False


def test_html_sanitization_strips_xss():
    from surya.inference.parsers import clean_block_html, parse_full_page_html

    dirty = (
        '<p onclick="alert(1)">Hello</p>'
        '<script>fetch("http://evil.tld")</script>'
        '<img src="x" onerror="alert(2)"/>'
        '<a href="javascript:alert(3)">Click</a>'
        '<a href="https://datalab.to">Safe</a>'
    )
    cleaned = clean_block_html(dirty)
    assert "<script" not in cleaned.lower()
    assert "onclick" not in cleaned.lower()
    assert "onerror" not in cleaned.lower()
    assert "javascript:" not in cleaned.lower()
    assert 'href="https://datalab.to"' in cleaned

    full_dirty = (
        '<div data-bbox="0 0 500 500" data-label="Text">'
        '<p onload="evil()">Safe text</p><iframe src="https://evil.tld"></iframe>'
        "</div>"
    )
    blocks = parse_full_page_html(full_dirty)
    assert len(blocks) == 1
    assert "<iframe" not in blocks[0].html.lower()
    assert "onload" not in blocks[0].html.lower()
    assert "Safe text" in blocks[0].html


def test_polygon_box_empty_input_raises_value_error():
    from surya.common.polygon import PolygonBox

    with pytest.raises(ValueError, match="Input must be either a bbox"):
        PolygonBox(polygon=[])


def test_parse_range_str_validation():
    import click
    from surya.scripts.config import CLILoader

    assert CLILoader.parse_range_str("0, 2-4, 2") == [0, 2, 3, 4]
    with pytest.raises(click.BadParameter):
        CLILoader.parse_range_str("5-2")
    with pytest.raises(click.BadParameter):
        CLILoader.parse_range_str("-1")


def test_top_level_package_exports():
    import surya

    assert surya.__version__ == "0.24.0"
    assert surya.SuryaInferenceManager is not None
    assert surya.LayoutPredictor is not None
    assert surya.RecognitionPredictor is not None
    assert surya.TableRecPredictor is not None


def test_parse_layout_compact_coord_in_bbox():
    from surya.inference.parsers import parse_layout

    blocks = parse_layout('[{"label": "Text", "bbox": "188 593 808<|670|>", "count": 60}]')
    assert len(blocks) == 1
    assert blocks[0].bbox == (188.0, 593.0, 808.0, 670.0)
    assert blocks[0].count == 60


def test_cli_skill_and_version_flags(tmp_path):
    from click.testing import CliRunner
    from surya.scripts.ocr_text import ocr_text_cli

    runner = CliRunner()
    rv = runner.invoke(ocr_text_cli, ["--version"])
    assert rv.exit_code == 0
    assert "0.24.0" in rv.output

    rs = runner.invoke(ocr_text_cli, ["--skill"])
    assert rs.exit_code == 0
    assert "name: surya" in rs.output

    ri = runner.invoke(ocr_text_cli, ["--install_skill", str(tmp_path)])
    assert ri.exit_code == 0
    assert (tmp_path / "surya" / "SKILL.md").is_file()


def test_viewer_engine_upload_and_modes():
    import io
    from PIL import Image
    from types import SimpleNamespace
    from surya.inference.schema import BatchOutputItem
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor
    from surya.scripts.viewer import ViewerEngine, render_viewer_html
    from surya.table_rec import TableRecPredictor

    html = render_viewer_html()
    assert "Surya 2.5 Viewer" in html
    assert "__LOGO__" not in html

    def fake_generate(batch):
        out = []
        for item in batch:
            if item.prompt_type == "layout":
                raw = '[{"label": "Table", "bbox": "50 50 950 950", "count": 50}]'
            elif item.prompt_type == "high_accuracy_bbox":
                raw = '<div data-bbox="50 50 950 950" data-label="Text"><p>Hello <math>x^2</math></p></div>'
            else:
                raw = "<table><tr><th>A</th><td>1</td></tr></table>"
            out.append(
                BatchOutputItem(
                    raw=raw,
                    token_count=10,
                    error=False,
                    metadata=item.metadata,
                    finish_reason="stop",
                )
            )
        return out

    fake_mgr = SimpleNamespace(generate=fake_generate)
    preds = {
        "manager": fake_mgr,
        "layout": LayoutPredictor(fake_mgr),
        "recognition": RecognitionPredictor(fake_mgr),
        "table_rec": TableRecPredictor(fake_mgr),
    }
    engine = ViewerEngine(predictor_supplier=lambda: preds)
    assert engine.info()["version"] == "0.24.0"

    from PIL import ImageDraw

    img = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(img).rectangle((20, 20, 280, 180), fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    up = engine.upload(buf.getvalue(), filename="sample.png")
    assert up["pages"] == 1

    for mode in ("full_page", "block_ocr", "layout", "table_rec"):
        res = engine.page(up["id"], 0, {"mode": mode, "dpi": 150})
        assert res["mode_used"] == mode
        assert len(res["blocks"]) == 1
        assert res["timings"]["total_ms"] >= 0
