import copy
from typing import List
import torch
from functools import lru_cache

import torch.nn.functional as F

from surya.common.polygon import PolygonBox


def clean_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        xs = [point[0] for point in box_obj.polygon]
        ys = [point[1] for point in box_obj.polygon]
        if max(xs) == min(xs) or max(ys) == min(ys):
            continue

        box = box_obj.bbox
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            if box == other_box:
                continue
            if (
                box[0] >= other_box[0]
                and box[1] >= other_box[1]
                and box[2] <= other_box[2]
                and box[3] <= other_box[3]
            ):
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def rescale_bbox(bbox, processor_size, image_size):
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_bbox = copy.deepcopy(bbox)
    new_bbox[0] = int(new_bbox[0] * width_scaler)
    new_bbox[1] = int(new_bbox[1] * height_scaler)
    new_bbox[2] = int(new_bbox[2] * width_scaler)
    new_bbox[3] = int(new_bbox[3] * height_scaler)
    return new_bbox


def expand_bbox(bbox, expansion_factor=0.01):
    expansion_low = 1 - expansion_factor
    expansion_high = 1 + expansion_factor
    return [
        bbox[0] * expansion_low,
        bbox[1] * expansion_low,
        bbox[2] * expansion_high,
        bbox[3] * expansion_high,
    ]

SCRIPT_TOKEN_MAPPING = {
    "latin": "<SCRIPT-LATIN>",
    "punctuation": "<SCRIPT-PUNCTUATION>",
    "cyrillic": "<SCRIPT-CYRILLIC>",
    "arabic": "<SCRIPT-ARABIC>",
    "chinese": "<SCRIPT-CHINESE>",
    "japanese": "<SCRIPT-JAPANESE>",
    "korean": "<SCRIPT-KOREAN>",
    "symbols": "<SCRIPT-SYMBOLS>",
    "greek": "<SCRIPT-GREEK>",
    "armenian": "<SCRIPT-ARMENIAN>",
    "hebrew": "<SCRIPT-HEBREW>",
    "devanagari": "<SCRIPT-DEVANAGARI>",
    "bengali": "<SCRIPT-BENGALI>",
    "gurmukhi": "<SCRIPT-GURMUKHI>",
    "gujarati": "<SCRIPT-GUJARATI>",
    "oriya": "<SCRIPT-ORIYA>",
    "tamil": "<SCRIPT-TAMIL>",
    "telugu": "<SCRIPT-TELUGU>",
    "kannada": "<SCRIPT-KANNADA>",
    "malayalam": "<SCRIPT-MALAYALAM>",
    "sinhala": "<SCRIPT-SINHALA>",
    "thai": "<SCRIPT-THAI>",
    "lao": "<SCRIPT-LAO>",
    "myanmar": "<SCRIPT-MYANMAR>",
    "georgian": "<SCRIPT-GEORGIAN>",
    "ethiopic": "<SCRIPT-ETHIOPIC>",
    "khmer": "<SCRIPT-KHMER>",
    "mongolian": "<SCRIPT-MONGOLIAN>",
    "math": "<SCRIPT-MATH>",
}

@lru_cache(maxsize=1)
def script_ranges():
    script_categories = {
        # Latin-based scripts (used by English, French, German, etc.)
        "latin": [
            (0x0041, 0x005A),  # Latin uppercase A-Z
            (0x0061, 0x007A),  # Latin lowercase a-z
            (0x0080, 0x00FF),  # Latin-1 Supplement
            (0x0100, 0x017F),  # Latin Extended-A
            (0x0180, 0x024F),  # Latin Extended-B
            (0x0250, 0x02AF),  # IPA Extensions
            (0x02B0, 0x02FF),  # Spacing Modifier Letters
            (0x0300, 0x036F),  # Combining Diacritical Marks
            (0x1E00, 0x1EFF),  # Latin Extended Additional
            (0x2C60, 0x2C7F),  # Latin Extended-C
            (0xA720, 0xA7FF),  # Latin Extended-D
        ],
        # Punctuation, universal characters, and general symbols
        "punctuation": [
            (0x0020, 0x0020),  # Space
            (0x0021, 0x002F),  # Basic punctuation and symbols
            (0x0030, 0x0039),  # Digits 0-9
            (0x003A, 0x0040),  # More punctuation and symbols
            (0x005B, 0x0060),  # More punctuation and symbols
            (0x007B, 0x007F),  # More punctuation and symbols
            (0x2000, 0x206F),  # General Punctuation
        ],
        # Cyrillic scripts (used by Russian, Ukrainian, etc.)
        "cyrillic": [
            (0x0400, 0x04FF),  # Cyrillic
            (0x0500, 0x052F),  # Cyrillic Supplement
        ],
        # Arabic scripts
        "arabic": [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
        ],
        # Chinese characters
        "chinese": [
            (0x4E00, 0x9FFF),  # Common CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Extension A
            (0x20000, 0x2A6DF),  # CJK Extension B
        ],
        # Japanese-specific scripts (excluding shared CJK)
        "japanese": [
            (0x3040, 0x30FF),  # Hiragana and Katakana
        ],
        # Korean-specific scripts
        "korean": [
            (0x1100, 0x11FF),  # Hangul Jamo
            (0x3130, 0x318F),  # Hangul Compatibility Jamo
            (0xAC00, 0xD7AF),  # Hangul Syllables
        ],
        # Various mathematical and technical symbols
        "symbols": [
            (0x2070, 0x209F),  # Superscripts and Subscripts
            (0x20A0, 0x20CF),  # Currency Symbols
            (0x2100, 0x214F),  # Letterlike Symbols
            (0x2150, 0x218F),  # Number Forms
            (0x2190, 0x21FF),  # Arrows
            (0x2200, 0x22FF),  # Mathematical Operators
            (0x2300, 0x23FF),  # Miscellaneous Technical
            (0x2500, 0x257F),  # Box Drawing
            (0x2580, 0x259F),  # Block Elements
            (0x25A0, 0x25FF),  # Geometric Shapes
            (0x2600, 0x26FF),  # Miscellaneous Symbols
            (0x2700, 0x27BF),  # Dingbats
            (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
            (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
            (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
            (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
        ],
        # Individual scripts for languages with unique writing systems
        "greek": [(0x0370, 0x03FF)],  # Greek and Coptic
        "armenian": [(0x0530, 0x058F)],  # Armenian
        "hebrew": [(0x0590, 0x05FF)],  # Hebrew
        "devanagari": [(0x0900, 0x097F)],  # Devanagari (Hindi, Sanskrit)
        "bengali": [(0x0980, 0x09FF)],  # Bengali
        "gurmukhi": [(0x0A00, 0x0A7F)],  # Gurmukhi (Punjabi)
        "gujarati": [(0x0A80, 0x0AFF)],  # Gujarati
        "oriya": [(0x0B00, 0x0B7F)],  # Oriya
        "tamil": [(0x0B80, 0x0BFF)],  # Tamil
        "telugu": [(0x0C00, 0x0C7F)],  # Telugu
        "kannada": [(0x0C80, 0x0CFF)],  # Kannada
        "malayalam": [(0x0D00, 0x0D7F)],  # Malayalam
        "sinhala": [(0x0D80, 0x0DFF)],  # Sinhala
        "thai": [(0x0E00, 0x0E7F)],  # Thai
        "lao": [(0x0E80, 0x0EFF)],  # Lao
        "myanmar": [(0x1000, 0x109F)],  # Myanmar
        "georgian": [(0x10A0, 0x10FF)],  # Georgian
        "ethiopic": [(0x1200, 0x137F)],  # Ethiopic
        "khmer": [(0x1780, 0x17FF)],  # Khmer
        "mongolian": [(0x1800, 0x18AF)],  # Mongolian
    }

    # Convert to a flat structure with character ranges
    flat_ranges = {}
    for category, ranges in script_categories.items():
        # Create a set of all characters in this category
        char_set = set()
        for start, end in ranges:
            char_set.update(range(start, end + 1))

        # Store the set in flat_ranges
        flat_ranges[category] = char_set

    return script_categories, flat_ranges

def get_top_scripts(text: str, max_scripts: int = 5):
    script_categories, flat_ranges = script_ranges()
    char_count = {category: 0 for category in script_categories.keys()}
    for char in text:
        for category, char_set in flat_ranges.items():
            if ord(char) in char_set:
                char_count[category] += 1
                break

    top_scripts = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
    top_scripts = [ts[0] for ts in top_scripts if ts[1] > 0]
    if "<math" in text:
        top_scripts.insert(0, "math")

    return top_scripts[:max_scripts]

def is_flash_attn_2_supported(device: str | torch.device) -> bool:
    if not torch.cuda.is_available():
        return False

    if "cuda" not in str(device):
        return False

    # Check CUDA version >= 12.0
    cuda_version_str = torch.version.cuda
    if cuda_version_str is None:
        return False
    cuda_version = tuple(map(int, cuda_version_str.split(".")))
    if cuda_version < (12, 0):
        return False

    # Check GPU compute capability (Ampere, Ada, Hopper GPUs)
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    if compute_capability < 8.0:
        return False

    return True


def pad_to_batch_size_repeat(tensor: torch.Tensor, batch_size: int):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    if pad_size < 0:
        return tensor

    # Repeat the last row pad_size times
    last_row = tensor[-1:].repeat(pad_size, 1, 1)

    # Concatenate original tensor with repeated last rows
    return torch.cat([tensor, last_row], dim=0)


def pad_to_batch_size(tensor: torch.Tensor, batch_size: int):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

    return F.pad(tensor, padding, mode="constant", value=0)
