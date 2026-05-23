import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
from PIL import Image, ImageDraw

from surya.detection import DetectionPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from surya.table_rec import TableRecPredictor
from surya.settings import settings

@pytest.fixture(scope="session")
def ocr_error_predictor() -> OCRErrorPredictor:
    ocr_error_predictor = OCRErrorPredictor()
    yield ocr_error_predictor
    del ocr_error_predictor


@pytest.fixture(scope="session")
def layout_predictor() -> LayoutPredictor:
    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    yield layout_predictor
    del layout_predictor


@pytest.fixture(scope="session")
def detection_predictor() -> DetectionPredictor:
    detection_predictor = DetectionPredictor()
    yield detection_predictor
    del detection_predictor


@pytest.fixture(scope="session")
def recognition_predictor() -> RecognitionPredictor:
    recognition_predictor = RecognitionPredictor(FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT))
    yield recognition_predictor
    del recognition_predictor


@pytest.fixture(scope="session")
def table_rec_predictor() -> TableRecPredictor:
    table_rec_predictor = TableRecPredictor()
    yield table_rec_predictor
    del table_rec_predictor


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


@pytest.fixture()
def test_image_tall():
    image = Image.new("RGB", (4096, 4096), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello World", fill="black", font_size=72)
    draw.text(
        (4000, 4000),
        "This is a sentence of text.\n\nNow it is a paragraph.\n\nA three-line one.",
        fill="black",
        font_size=24,
    )
    return image

@pytest.fixture()
def test_image_latex():
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    img_path = os.path.join(assets_dir, "test_latex.png")
    image = Image.open(img_path).convert("RGB")
    return image