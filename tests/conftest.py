import os
import pytest
from PIL import Image, ImageDraw

from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings
from surya.table_rec import TableRecPredictor


@pytest.fixture(scope="session")
def manager() -> SuryaInferenceManager:
    """Eagerly start the VLM backend. If the runner has neither vllm nor
    llama-server available (e.g. GitHub Actions ubuntu / windows runners),
    skip every VLM-dependent test in this session instead of failing them."""
    if "SURYA_INFERENCE_STARTUP_TIMEOUT" not in os.environ:
        settings.SURYA_INFERENCE_STARTUP_TIMEOUT = 30.0
    m = SuryaInferenceManager(lazy=True)
    try:
        m.start()
    except Exception as exc:  # SpawnError, binary missing, port issues, etc.
        pytest.skip(f"VLM backend unavailable in this environment: {exc}")
    yield m
    try:
        m.stop()
    except Exception:
        pass


@pytest.fixture(scope="session")
def layout_predictor(manager) -> LayoutPredictor:
    return LayoutPredictor(manager)


@pytest.fixture(scope="session")
def recognition_predictor(manager) -> RecognitionPredictor:
    return RecognitionPredictor(manager)


@pytest.fixture(scope="session")
def table_rec_predictor(manager) -> TableRecPredictor:
    return TableRecPredictor(manager)


@pytest.fixture()
def test_image():
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello World", fill="black", font_size=72)
    draw.text(
        (10, 200),
        "This is a sentence of text.\nNow it is a paragraph.\nA three-line one.",
        fill="black",
        font_size=24,
    )
    return image
