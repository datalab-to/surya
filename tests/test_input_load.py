import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_load_pdf_empty_page_range_returns_no_pages(monkeypatch):
    fake_pil = types.ModuleType("PIL")
    fake_pil.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})

    fake_image_module = types.ModuleType("PIL.Image")
    fake_image_module.open = MagicMock()
    fake_pil.Image = fake_image_module

    fake_processing = types.ModuleType("surya.input.processing")
    fake_processing.open_pdf = MagicMock()
    fake_processing.get_page_images = MagicMock()

    fake_settings_module = types.ModuleType("surya.settings")
    fake_settings_module.settings = SimpleNamespace(IMAGE_DPI=96)

    fake_logging = types.ModuleType("surya.logging")
    fake_logging.get_logger = lambda: MagicMock()

    fake_filetype = types.ModuleType("filetype")
    fake_filetype.guess = MagicMock(return_value=None)

    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)
    monkeypatch.setitem(sys.modules, "filetype", fake_filetype)
    monkeypatch.setitem(sys.modules, "surya.input.processing", fake_processing)
    monkeypatch.setitem(sys.modules, "surya.settings", fake_settings_module)
    monkeypatch.setitem(sys.modules, "surya.logging", fake_logging)
    monkeypatch.delitem(sys.modules, "surya.input.load", raising=False)

    load = importlib.import_module("surya.input.load")

    doc = MagicMock()
    doc.__len__.return_value = 3
    fake_processing.open_pdf.return_value = doc

    images, names = load.load_pdf("test.pdf", page_range=[])

    assert images == []
    assert names == []
    fake_processing.get_page_images.assert_not_called()
    doc.close.assert_called_once()
