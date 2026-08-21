"""Public entry point for figure/diagram description via InternVL3 (CPU-only).

    from surya.figure_caption import analyze_image
    result = analyze_image("page3_diagram.png", "Describe this image in detail.")
    # {"success": True, "response": "...", "error": None}
"""

import torch
from PIL import Image, UnidentifiedImageError

from surya.figure_caption.model import get_model, image_to_pixel_values
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

__all__ = ["analyze_image"]


def analyze_image(image_path: str, prompt: str) -> dict:
    """Run one InternVL3 vision-language query against a single image.

    Returns {"success": bool, "response": str | None, "error": str | None}.
    Never raises — failures (missing file, corrupt image, model/inference
    errors) are reported through the "error" field so callers processing many
    figures in a batch don't need to wrap every call in its own try/except.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return _failure(f"Image not found: {image_path}")
    except UnidentifiedImageError:
        return _failure(f"Not a readable image file: {image_path}")
    except OSError as e:
        return _failure(f"Could not open image {image_path}: {e}")

    try:
        model, tokenizer = get_model()
    except Exception as e:
        logger.error(f"Failed to load figure-captioning model: {e}")
        return _failure(f"Model load failed: {e}")

    try:
        pixel_values = image_to_pixel_values(image)
        question = f"<image>\n{prompt}"
        generation_config = dict(
            max_new_tokens=settings.FIGURE_CAPTION_MAX_NEW_TOKENS,
            do_sample=False,
        )
        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, question, generation_config)
    except Exception as e:
        logger.error(f"InternVL3 inference failed for {image_path}: {e}")
        return _failure(f"Inference failed: {e}")

    return {"success": True, "response": response, "error": None}


def _failure(message: str) -> dict:
    return {"success": False, "response": None, "error": message}
