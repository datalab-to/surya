from typing import List
import PIL

from surya.input.processing import open_pdf, get_page_images
from surya.logging import get_logger
from surya.settings import settings
import os
import filetype
from PIL import Image
import json

logger = get_logger()


def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]


def load_pdf(pdf_path, page_range: List[int] | None = None, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_path)
    last_page = len(doc)

    if page_range is not None:
        if len(page_range) == 0:
            doc.close()
            return [], []
        assert all([0 <= page < last_page for page in page_range]), (
            f"Invalid page range: {page_range}"
        )
    else:
        page_range = list(range(last_page))

    images = get_page_images(doc, page_range, dpi=dpi)
    doc.close()
    names = [get_name_from_path(pdf_path) for _ in page_range]
    return images, names


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    name = get_name_from_path(image_path)
    return [image], [name]


def load_from_file(
    input_path, page_range: List[int] | None = None, dpi=settings.IMAGE_DPI
):
    input_type = filetype.guess(input_path)
    if input_type and input_type.extension == "pdf":
        return load_pdf(input_path, page_range, dpi=dpi)
    else:
        return load_image(input_path)


def load_from_folder(
    folder_path, page_range: List[int] | None = None, dpi=settings.IMAGE_DPI
):
    image_paths = [
        os.path.join(folder_path, image_name)
        for image_name in os.listdir(folder_path)
        if not image_name.startswith(".")
    ]
    image_paths = [ip for ip in image_paths if not os.path.isdir(ip)]

    images = []
    names = []
    for path in image_paths:
        extension = filetype.guess(path)
        if extension and extension.extension == "pdf":
            image, name = load_pdf(path, page_range, dpi=dpi)
            images.extend(image)
            names.extend(name)
        else:
            try:
                image, name = load_image(path)
                images.extend(image)
                names.extend(name)
            except PIL.UnidentifiedImageError:
                logger.warning(f"Could not load image {path}")
                continue
    return images, names


def load_lang_file(lang_path, names):
    with open(lang_path, "r") as f:
        lang_dict = json.load(f)
    return [lang_dict[name].copy() for name in names]
