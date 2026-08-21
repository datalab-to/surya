"""CPU-only InternVL3 singleton for general-purpose figure/diagram description.

surya's own VLM (surya/inference/) is fine-tuned only for layout detection,
OCR, and table recognition — as confirmed empirically, it ignores any other
instruction and always emits one of those three JSON schemas regardless of
the prompt. This module is a separate, general vision-language model used
only to describe non-text regions (Figure/Picture/Diagram/...) that surya's
own VLM never runs OCR on in the first place.

Loaded once per process (see get_model()) since InternVL3-1B is a few hundred
MB of weights in float32 — not something to reload per call.
"""

import threading

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_lock = threading.Lock()
_model = None
_tokenizer = None


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num: int, max_num: int, image_size: int, use_thumbnail: bool):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    tiles_per_row = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def image_to_pixel_values(image: Image.Image, input_size: int = None, max_num: int = None) -> torch.Tensor:
    """PIL image -> tiled pixel_values tensor, CPU float32, ready for model.chat()."""
    input_size = input_size or settings.FIGURE_CAPTION_INPUT_SIZE
    max_num = max_num or settings.FIGURE_CAPTION_MAX_TILES

    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=True)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values.to(torch.float32)


_post_init_patch_applied = False


def _patch_missing_all_tied_weights_keys() -> None:
    """Work around a transformers v5 incompatibility in InternVL's remote code.

    transformers v5 expects every PreTrainedModel subclass to call
    self.post_init() at the end of __init__ (it's what sets up
    all_tied_weights_keys, read later during from_pretrained's weight-tying
    step: modeling_utils.py's _move_missing_keys_from_meta_to_device does
    `missing_keys - self.all_tied_weights_keys.keys()`). InternVL's
    InternVLChatModel.__init__ predates that contract and never calls
    post_init(), which raises
    "'InternVLChatModel' object has no attribute 'all_tied_weights_keys'"
    partway through from_pretrained. This is a known, widespread break across
    many trust_remote_code models on transformers v5, not specific to
    InternVL — see e.g. huggingface/transformers#43883, #43957.

    Pre-patching InternVLChatModel.__init__ itself doesn't work here: despite
    Python's sys.modules import cache, from_pretrained's own trust_remote_code
    class resolution ends up with a distinct class object, so a patched
    __init__ set beforehand is never called. Patching the one call site in
    transformers.modeling_utils instead sidesteps that entirely, and is safe
    for every other model (surya's own included) because it only fills in a
    default when the attribute is genuinely missing — models that set it up
    correctly via their own post_init() are untouched.
    """
    global _post_init_patch_applied
    if _post_init_patch_applied:
        return

    from transformers import modeling_utils

    original = modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device

    def patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original(self, *args, **kwargs)

    modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = patched
    _post_init_patch_applied = True


def get_model():
    """Lazily load the InternVL3 model + tokenizer once, CPU/float32 only."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    with _lock:
        if _model is not None:  # re-check inside the lock
            return _model, _tokenizer

        checkpoint = settings.FIGURE_CAPTION_MODEL_CHECKPOINT
        logger.info(f"Loading figure-captioning model {checkpoint} (CPU, float32)...")

        if settings.FIGURE_CAPTION_NUM_THREADS:
            torch.set_num_threads(settings.FIGURE_CAPTION_NUM_THREADS)

        _patch_missing_all_tied_weights_keys()

        model = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        model = model.eval().to("cpu")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)

        _model, _tokenizer = model, tokenizer
        logger.info("Figure-captioning model loaded.")

    return _model, _tokenizer
