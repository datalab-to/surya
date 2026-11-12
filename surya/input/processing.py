from typing import List
import pypdfium2

from surya.settings import settings


def open_pdf(pdf_filepath):
    return pypdfium2.PdfDocument(pdf_filepath)


def get_page_images(doc, indices: List[int], dpi: int = settings.IMAGE_DPI):
    images = []
    scale = dpi / 72
    for i in indices:
        page = doc[i]
        try:
            bitmap = page.render(scale=scale, draw_annots=False)
            try:
                images.append(bitmap.to_pil().convert("RGB"))
            finally:
                bitmap.close()
        finally:
            page.close()
    return images
