import importlib
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def load_input_module(monkeypatch):
    fake_pil = types.ModuleType("PIL")
    fake_pil.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})

    fake_image_module = types.ModuleType("PIL.Image")
    fake_image_module.open = MagicMock()
    fake_pil.Image = fake_image_module

    fake_processing = types.ModuleType("surya.input.processing")
    fake_processing.open_pdf = MagicMock()
    fake_processing.get_page_images = MagicMock()

    fake_settings = types.ModuleType("surya.settings")
    fake_settings.settings = SimpleNamespace(IMAGE_DPI=96)

    fake_logging = types.ModuleType("surya.logging")
    fake_logging.get_logger = lambda: MagicMock()

    fake_filetype = types.ModuleType("filetype")
    fake_filetype.guess = MagicMock(return_value=None)

    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)
    monkeypatch.setitem(sys.modules, "surya.input.processing", fake_processing)
    monkeypatch.setitem(sys.modules, "surya.settings", fake_settings)
    monkeypatch.setitem(sys.modules, "surya.logging", fake_logging)
    monkeypatch.setitem(sys.modules, "filetype", fake_filetype)
    monkeypatch.delitem(sys.modules, "surya.input.load", raising=False)

    return importlib.import_module("surya.input.load")


def load_config_module(monkeypatch):
    fake_load = types.ModuleType("surya.input.load")
    fake_load.load_from_file = lambda filepath, page_range, dpi=96: (
        [object()],
        ["paper.v1"],
    )
    fake_load.load_from_folder = lambda filepath, page_range, dpi=96: ([], [])

    fake_settings = types.ModuleType("surya.settings")
    fake_settings.settings = SimpleNamespace(
        RESULT_DIR="results", IMAGE_DPI_HIGHRES=192
    )

    monkeypatch.setitem(sys.modules, "surya.input.load", fake_load)
    monkeypatch.setitem(sys.modules, "surya.settings", fake_settings)
    monkeypatch.delitem(sys.modules, "surya.scripts.config", raising=False)

    return importlib.import_module("surya.scripts.config")


def test_get_name_from_path_preserves_multi_dot_stem(monkeypatch):
    load = load_input_module(monkeypatch)

    assert load.get_name_from_path("C:/tmp/paper.v1.pdf") == "paper.v1"


def test_cli_loader_result_path_preserves_multi_dot_stem(monkeypatch):
    config = load_config_module(monkeypatch)
    output_dir = Path(__file__).with_name("_tmp_results")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        loader = config.CLILoader("paper.v1.pdf", {"output_dir": str(output_dir)})
        assert Path(loader.result_path).name == "paper.v1"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
