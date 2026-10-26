"""A response cut by the token cap must never come back as a clean full page (no GPU: fake manager)."""

from types import SimpleNamespace

from PIL import Image

from surya.inference.schema import BatchOutputItem
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor

PAGE = "<|Text|><|10|><|20|><|900|><|300|><p>full page text</p>"


def _manager(full_page_finish, page=PAGE):
    def generate(batch):
        outs = []
        for it in batch:
            if it.prompt_type == "high_accuracy_bbox":
                outs.append(
                    BatchOutputItem(
                        raw=page,
                        token_count=9,
                        error=False,
                        metadata=it.metadata,
                        finish_reason=full_page_finish,
                    )
                )
            else:
                outs.append(
                    BatchOutputItem(
                        raw="<p>block text</p>",
                        token_count=3,
                        error=False,
                        metadata=it.metadata,
                        finish_reason="stop",
                    )
                )
        return outs

    return SimpleNamespace(generate=generate)


def _layout(w, h):
    box = LayoutBox(
        polygon=[[10, 10], [w - 10, 10], [w - 10, h - 10], [10, h - 10]],
        label="Text",
        raw_label="Text",
        position=0,
    )
    return LayoutResult(bboxes=[box], image_bbox=[0, 0, w, h])


def _img():
    from PIL import ImageDraw

    im = Image.new("RGB", (400, 500), "white")
    d = ImageDraw.Draw(im)
    for y in range(40, 460, 14):
        d.text((30, y), "some dark text on the page", fill=(0, 0, 0))
    return im


def test_length_finish_routes_to_block_fallback():
    r = RecognitionPredictor(_manager("length"))(
        [_img()], [_layout(400, 500)], full_page=True
    )[0]
    assert "block text" in " ".join(
        b.html for b in r.blocks
    ) and "full page text" not in " ".join(b.html for b in r.blocks)


def test_clean_stop_keeps_full_page():
    r = RecognitionPredictor(_manager("stop"))(
        [_img()], [_layout(400, 500)], full_page=True
    )[0]
    assert "full page text" in " ".join(b.html for b in r.blocks)


def test_unparseable_nonblank_output_routes_to_block_fallback():
    r = RecognitionPredictor(_manager("stop", page="<|Text|><|10|><|20|>"))(
        [_img()], [_layout(400, 500)], full_page=True
    )[0]
    assert "block text" in " ".join(b.html for b in r.blocks)
