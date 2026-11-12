from pathlib import Path
from typing import List
import os

import filetype
import PIL
from PIL import Image

from surya.input.processing import open_pdf, get_page_images
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

SUPPORTED_FILE_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
}


def get_name_from_path(path) -> str:
    return Path(path).stem


def discover_input_files(folder_path: str, recursive: bool = False) -> List[str]:
    """Discover supported PDF and image files in a folder in deterministic order."""
    folder_path = os.path.abspath(folder_path)
    if os.path.isfile(folder_path):
        return [folder_path]
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input path does not exist: {folder_path}")

    discovered: List[str] = []
    if recursive:
        for root, dirs, filenames in os.walk(folder_path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_FILE_EXTENSIONS:
                    full = os.path.join(root, fname)
                    if os.path.isfile(full):
                        discovered.append(full)
    else:
        for fname in sorted(os.listdir(folder_path)):
            if fname.startswith("."):
                continue
            full = os.path.join(folder_path, fname)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_FILE_EXTENSIONS:
                discovered.append(full)
    return discovered


def load_pdf(pdf_path, page_range: List[int] | None = None, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_path)
    try:
        last_page = len(doc)

        if page_range is not None:
            if not page_range:
                return [], []
            if not all(0 <= page < last_page for page in page_range):
                raise ValueError(
                    f"Invalid page range {page_range} for PDF with {last_page} page(s)."
                )
        else:
            page_range = list(range(last_page))

        images = get_page_images(doc, page_range, dpi=dpi)
    finally:
        doc.close()
    names = [get_name_from_path(pdf_path) for _ in page_range]
    return images, names


def load_image(image_path):
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    name = get_name_from_path(image_path)
    return [image], [name]


def load_from_file(
    input_path, page_range: List[int] | None = None, dpi=settings.IMAGE_DPI
):
    input_type = filetype.guess(input_path)
    if (input_type and input_type.extension == "pdf") or str(input_path).lower().endswith(".pdf"):
        return load_pdf(input_path, page_range, dpi=dpi)
    return load_image(input_path)


def load_from_folder(
    folder_path,
    page_range: List[int] | None = None,
    dpi=settings.IMAGE_DPI,
    recursive: bool = False,
):
    image_paths = discover_input_files(folder_path, recursive=recursive)

    images = []
    names = []
    for path in image_paths:
        extension = filetype.guess(path)
        if (extension and extension.extension == "pdf") or path.lower().endswith(".pdf"):
            try:
                image, name = load_pdf(path, page_range, dpi=dpi)
                images.extend(image)
                names.extend(name)
            except Exception as e:
                logger.warning(f"Could not load PDF {path}: {e}")
                continue
        else:
            try:
                image, name = load_image(path)
                images.extend(image)
                names.extend(name)
            except (PIL.UnidentifiedImageError, OSError) as e:
                logger.warning(f"Could not load image {path}: {e}")
                continue
    return images, names
