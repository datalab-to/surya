from types import SimpleNamespace
import pytest
from PIL import Image, ImageDraw

from surya.inference.schema import BatchOutputItem
from surya.table_rec import TableRecPredictor


def test_table_rec(table_rec_predictor):
    data = [
        ["Name", "Age", "City"],
        ["Alice", 25, "New York"],
        ["Bob", 30, "Los Angeles"],
        ["Charlie", 35, "Chicago"],
    ]
    test_image = draw_table(data)

    results = table_rec_predictor([test_image])
    assert len(results) == 1
    assert results[0].image_bbox == [0, 0, test_image.size[0], test_image.size[1]]
    if results[0].error:
        return
    assert results[0].mode == "full"
    assert isinstance(results[0].html, str)
    assert "<table" in results[0].html.lower()


def test_table_rec_unit_mocked():
    fake_manager = SimpleNamespace(
        generate=lambda batch: [
            BatchOutputItem(
                raw="<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
                token_count=20,
                error=False,
                finish_reason="stop",
            )
            for _ in batch
        ]
    )
    pred = TableRecPredictor(fake_manager)
    img = draw_table([["A", "B"], [1, 2]])
    res = pred([img])
    assert len(res) == 1
    assert not res[0].error
    assert "<table>" in res[0].html


def test_table_rec_rejects_file_paths():
    pred = TableRecPredictor()
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        pred(["path/to/table.png"])


def draw_table(data, cell_width=100, cell_height=40):
    rows = len(data)
    cols = len(data[0])
    width = cols * cell_width
    height = rows * cell_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for i in range(rows + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill="black", width=1)

    for i in range(cols + 1):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill="black", width=1)

    for i in range(rows):
        for j in range(cols):
            text = str(data[i][j])
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = j * cell_width + (cell_width - text_width) // 2
            y = i * cell_height + (cell_height - text_height) // 2

            draw.text((x, y), text, fill="black")

    return image
