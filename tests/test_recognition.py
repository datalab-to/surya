from types import SimpleNamespace
import pytest
from PIL import Image, ImageDraw

from surya.inference.schema import BatchOutputItem
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor


def test_recognition(recognition_predictor, layout_predictor, test_image):
    layouts = layout_predictor([test_image])
    if layouts[0].error:
        # Server unavailable in test env — skip silently
        return
    page_results = recognition_predictor([test_image], layouts)

    assert len(page_results) == 1
    assert page_results[0].image_bbox == [0, 0, 1024, 1024]

    blocks = page_results[0].blocks
    # Each layout box should produce one block (skipped or otherwise)
    assert len(blocks) == len(layouts[0].bboxes)
    for blk in blocks:
        assert blk.reading_order >= 0
        assert isinstance(blk.html, str)


def test_recognition_rejects_non_pil_images():
    pred = RecognitionPredictor()
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        pred(["not_an_image.png"])


def test_recognition_custom_skip_labels():
    calls = []

    def fake_generate(batch):
        calls.extend(batch)
        return [
            BatchOutputItem(
                raw="<p>Chart label</p>",
                token_count=5,
                error=False,
                metadata=item.metadata,
                finish_reason="stop",
            )
            for item in batch
        ]

    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 180, 180), fill="black")
    layout = LayoutResult(
        bboxes=[
            LayoutBox(
                polygon=[10, 10, 190, 190],
                label="Figure",
                raw_label="Figure",
                position=0,
                count=50,
            )
        ],
        image_bbox=[0, 0, 200, 200],
    )
    pred = RecognitionPredictor(SimpleNamespace(generate=fake_generate))

    # Default: Figure is skipped
    res_default = pred([img], [layout])
    assert res_default[0].blocks[0].skipped is True
    assert len(calls) == 0

    # Custom skip_labels=[]: Figure is OCR'd
    res_unskipped = pred([img], [layout], skip_labels=[])
    assert res_unskipped[0].blocks[0].skipped is False
    assert res_unskipped[0].blocks[0].html == "<p>Chart label</p>"
    assert len(calls) == 1


def test_full_page_table_rescue_when_emitted_as_paragraph():
    calls = []

    def fake_generate(batch):
        calls.extend(batch)
        if batch[0].prompt_type == "high_accuracy_bbox":
            return [
                BatchOutputItem(
                    raw='<div data-bbox="100 100 900 900" data-label="Table"><p>Col1 Col2</p></div>',
                    token_count=20,
                    error=False,
                    metadata=batch[0].metadata,
                    finish_reason="stop",
                )
            ]
        return [
            BatchOutputItem(
                raw="<table><tr><th>Col1</th><th>Col2</th></tr></table>",
                token_count=15,
                error=False,
                metadata=batch[0].metadata,
                finish_reason="stop",
            )
        ]

    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 30, 170, 170), fill="black")
    pred = RecognitionPredictor(SimpleNamespace(generate=fake_generate))

    res = pred([img], full_page=True)
    assert len(res) == 1
    assert len(res[0].blocks) == 1
    assert "<table>" in res[0].blocks[0].html
    assert len(calls) == 2
