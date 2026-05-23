from PIL import ImageDraw, ImageFont

from surya.debug.fonts import get_font_path
from surya.debug.text import get_text_size


def draw_bboxes_on_image(
    bboxes, image, labels=None, label_font_size=10, color: str | list = "red"
):
    polys = []
    for bb in bboxes:
        # Clockwise polygon
        poly = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
        polys.append(poly)

    return draw_polys_on_image(
        polys, image, labels, label_font_size=label_font_size, color=color
    )


def draw_polys_on_image(
    corners,
    image,
    labels=None,
    box_padding=-1,
    label_offset=1,
    label_font_size=10,
    color: str | list = "red",
):
    draw = ImageDraw.Draw(image)
    font_path = get_font_path()
    label_font = ImageFont.truetype(font_path, label_font_size)

    for i in range(len(corners)):
        poly = corners[i]
        poly = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(
            poly, outline=color[i] if isinstance(color, list) else color, width=1
        )

        if labels is not None:
            label = labels[i]
            text_position = (
                min([p[0] for p in poly]) + label_offset,
                min([p[1] for p in poly]) + label_offset,
            )
            text_size = get_text_size(label, label_font)
            box_position = (
                text_position[0] - box_padding + label_offset,
                text_position[1] - box_padding + label_offset,
                text_position[0] + text_size[0] + box_padding + label_offset,
                text_position[1] + text_size[1] + box_padding + label_offset,
            )
            try:
                draw.rectangle(box_position, fill="white")
            except Exception as e:
                print(f"Error drawing rectangle at {box_position}: {e}")
                continue
            draw.text(
                text_position,
                label,
                fill=color[i] if isinstance(color, list) else color,
                font=label_font,
            )

    return image
