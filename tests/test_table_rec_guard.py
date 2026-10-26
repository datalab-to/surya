"""predict_simple must flag an answer without Row/Col entries as an error (no GPU)."""

from types import SimpleNamespace

from PIL import Image

from surya.table_rec import TableRecPredictor


def _predict(raw):
    manager = SimpleNamespace(
        generate=lambda batch: [SimpleNamespace(error=False, raw=raw) for _ in batch]
    )
    return TableRecPredictor(manager).predict_simple(
        [Image.new("RGB", (100, 100), "white")]
    )[0]


def test_layout_style_answer_is_an_error():
    r = _predict('[{"label": "Table", "bbox": "0 0 1000 1000", "count": 760}]')
    assert r.error and not r.rows and not r.cols


def test_row_col_answer_is_ok():
    r = _predict(
        '[{"label": "Row", "bbox": "0 0 1000 500"}, {"label": "Col", "bbox": "0 0 500 1000"}]'
    )
    assert not r.error and len(r.rows) == 1 and len(r.cols) == 1 and len(r.cells) == 1


def test_cli_defaults_to_full_mode():
    from surya.scripts.table_recognition import table_recognition_cli

    mode = next(p for p in table_recognition_cli.params if p.name == "mode")
    assert mode.default == "full"
