import math

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from typing import List, Optional, Tuple

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.processor.schema import (
    TextInput,
    ImageInput,
    ProcessorOutput,
)
from surya.common.surya.schema import TaskNames
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

# Task agnostic tokens - Every task will use these in some form or another
EOS_TOKEN = "</S>"
EOI_TOKEN = "<EOI>"  # This is end of INPUT, not image. Images are always followed by a task specific BOS token, so that serves as a delimiter anyways.
IMAGE_TOKEN = "<IMAGE>"
PAD_TOKEN = "<PAD>"
NO_OUTPUT_TOKEN = "<NOP>"
IMAGE_ROTATED_TOKEN = "<ROT>"
REGISTER_TOKENS = ["<REG1>", "<REG2>", "<REG3>", "<REG4>"]
BEACON_TOKEN = "<BEACON>"
NOMATH_TOKEN = "<NO-MATH>"

# Task specific tokens
OCR_WITH_BOXES_BOS_TOKEN = "<OCR-WB>"
OCR_WITHOUT_BOXES_BOS_TOKEN = "<OCR-WOB>"
BLOCK_WITHOUT_BOXES_TOKEN = "<BLOCKS-WOB>"
LAYOUT_BOS_TOKEN = "<LAYOUT>"
TABLE_STRUCTURE_BOS_TOKEN = "<TABLE-STRUCTURE>"


class SuryaOCRProcessor(S3DownloaderMixin, ProcessorMixin):
    attributes = ["image_processor", "ocr_tokenizer"]
    image_processor_class = "BaseImageProcessor"
    ocr_tokenizer_class = "PreTrainedTokenizer"
    rescale_factor = 1 / 255.0
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        ocr_tokenizer: PreTrainedTokenizer,
        blank_bbox_token_id: int,
        num_register_tokens: int,
        patch_size: int,
        merge_size: int,
        num_beacon_tokens: int,
        beacon_token_interval: int,
        model_device: str,
        **kwargs,
    ):
        self.ocr_tokenizer = ocr_tokenizer
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.num_register_tokens = num_register_tokens
        self.num_beacon_tokens = num_beacon_tokens
        self.beacon_token_interval = beacon_token_interval

        self.tokenizer_vocab_size = 0
        for attr in self.attributes:
            if "tokenizer" in attr:
                self.tokenizer_vocab_size += getattr(self, attr).vocab_size

        self.offsets = {"ocr": 0}

        # Create special token mapping
        self.special_token_mapping = self.ocr_tokenizer.system_tokens

        self.register_token_ids = [
            self.special_token_mapping.get(r) for r in REGISTER_TOKENS
        ]
        self.beacon_token_id = self.special_token_mapping.get(BEACON_TOKEN)
        self.image_token_id = self.special_token_mapping.get(IMAGE_TOKEN)
        self.pad_token_id = self.special_token_mapping.get(PAD_TOKEN)
        self.eos_token_id = self.special_token_mapping.get(EOS_TOKEN)
        self.eoi_token_id = self.special_token_mapping.get(EOI_TOKEN)
        self.no_output_token = self.special_token_mapping.get(NO_OUTPUT_TOKEN)
        self.image_rotated_token = self.special_token_mapping.get(IMAGE_ROTATED_TOKEN)
        self.nomath_token = self.special_token_mapping.get(NOMATH_TOKEN)

        self.bos_token_id = {
            TaskNames.ocr_with_boxes: self.special_token_mapping.get(
                OCR_WITH_BOXES_BOS_TOKEN
            ),
            TaskNames.ocr_without_boxes: self.special_token_mapping.get(
                OCR_WITHOUT_BOXES_BOS_TOKEN
            ),
            TaskNames.block_without_boxes: self.special_token_mapping.get(
                BLOCK_WITHOUT_BOXES_TOKEN
            ),
            TaskNames.layout: self.special_token_mapping.get(LAYOUT_BOS_TOKEN),
            TaskNames.table_structure: self.special_token_mapping.get(
                TABLE_STRUCTURE_BOS_TOKEN
            ),
        }

        if self.image_token_id is None:
            logger.warning("Warning: Image token not found in special tokens")

        self.blank_bbox_token_id = blank_bbox_token_id
        self.bbox_pad_token_id = self.blank_bbox_token_id

        self.ignore_bbox_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k not in self.ocr_tokenizer.special_tokens["math_external"]
        ]
        math_end_token = "</math>"
        self.math_start_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k in self.ocr_tokenizer.special_tokens["math_external"]
            and k != math_end_token
        ]
        self.math_end_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k == math_end_token
        ]

        if self.num_register_tokens > len(self.register_token_ids):
            raise ValueError(
                "The number of register tokens requested exceeds the number of register tokens defined in the special token mapping."
            )

        self.image_mean = np.array(self.image_mean, dtype=np.float32)
        self.image_std = np.array(self.image_std, dtype=np.float32)
        self.model_device = model_device

    @property
    def vocab_size(self):
        return self.tokenizer_vocab_size

    def image_processor(self, image: Image.Image) -> np.ndarray:
        # Convert to array
        image = np.asarray(image, dtype=np.float32)
        return image

    @staticmethod
    def scale_to_fit(
        img: np.ndarray,
        max_size: Tuple[int, int],
        min_size: Tuple[int, int] = (168, 168),
    ):
        # Get current dimensions
        height, width = img.shape[:2]

        # Check for empty or invalid image
        if width == 0 or height == 0:
            return img

        max_width, max_height = max_size
        min_width, min_height = min_size

        # Calculate pixel counts
        current_pixels = width * height
        max_pixels = max_width * max_height
        min_pixels = min_width * min_height

        if current_pixels > max_pixels:
            scale_factor = (max_pixels / current_pixels) ** 0.5

            new_width = math.floor(width * scale_factor)
            new_height = math.floor(height * scale_factor)
        elif current_pixels == 0:
            return img
        elif current_pixels < min_pixels:
            scale_factor = (min_pixels / current_pixels) ** 0.5

            new_width = math.ceil(width * scale_factor)
            new_height = math.ceil(height * scale_factor)
        else:
            return img

        return cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )

    def _image_processor(self, image: np.ndarray):
        image = image.astype(np.float64) * self.rescale_factor
        image = (image.astype(np.float32) - self.image_mean) / self.image_std
        return image

    def _process_and_tile(
        self, image: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Resizes the input image to the closest multiple of tile_size while preserving the aspect ratio
        and returns a tensor of image tiles.
        """
        extra_multipler = (
            4 if settings.FOUNDATION_XLA else 1
        )  # Needed to force same size grid_thws per row with padding

        factor = (
            self.patch_size * self.merge_size * extra_multipler
        )  # Make a multiple of window size

        height, width = image.shape[:2]

        h_bar = math.ceil(height / factor) * factor
        w_bar = math.ceil(width / factor) * factor
        if h_bar != height or w_bar != width:
            if height == 0 or width == 0:
                image = np.zeros((h_bar, w_bar, 3), dtype=np.uint8)
            else:
                image = cv2.resize(image, (w_bar, h_bar), interpolation=cv2.INTER_CUBIC)

        # Handle scaling and normalization
        image = self._image_processor(image)
        height, width = image.shape[:2]

        # Numpy array to torch tensor
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        patches = img_tensor.unsqueeze(0)

        channel = patches.shape[1]
        grid_t = patches.shape[0]
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        patches = patches.reshape(
            grid_t,
            1,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * 1 * self.patch_size * self.patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    # Handle image input dictionaries - Process image, tile accordingly, and setup the input ids and boxes correspondingly
    def _process_image_input(self, image_input: ImageInput) -> ProcessorOutput:
        rotated = image_input.get("rotated", False)
        image = image_input.get("image", None)

        assert image is not None, (
            "A PIL Image must be provided when the input type is `image`"
        )
        image_tiles, grid_thw = self._process_and_tile(image)

        num_tokens = image_tiles.shape[0] / self.merge_size**2
        assert num_tokens.is_integer(), (
            f"Expected number of tokens to be an integer, got {num_tokens}"
        )

        input_ids = [self.image_token_id] * int(num_tokens)
        input_ids += self.register_token_ids[: self.num_register_tokens]

        # Handle the image being rotated in the imdataset
        if rotated:
            input_ids = [self.image_rotated_token] + input_ids

        return ProcessorOutput(
            input_ids=input_ids,
            image_tiles=image_tiles,
            grid_thw=grid_thw,
        )

    def _process_text_input(self, text_input: TextInput, task: str) -> ProcessorOutput:
        input_text = text_input.get("text", None)
        math_mode = text_input.get("math", False)

        input_ids = self.ocr_tokenizer(input_text, tasks=task)["input_ids"][0]
        input_ids = [self.offsets["ocr"] + id for id in input_ids]

        # nomath token does not work for layout
        if not math_mode and task != "layout":
            input_ids.insert(0, self.nomath_token)

        return ProcessorOutput(
            input_ids=input_ids,
            image_tiles=None,
            grid_thw=None,
        )

    def _process_input(self, input_dict: dict, task: str):
        input_type = input_dict["type"]
        if input_type == "image":
            return self._process_image_input(input_dict)
        elif input_type == "text":
            return self._process_text_input(input_dict, task)

        raise NotImplementedError(f"Input of type `{input_type}` is not implemented")

    # Peprocessing for OCR task
    # The task is expected to have - image_dict, user_input_dict, output_dict
    # use_input_dict is allowed to have an empty input which is fine, but needs to be present
    def _process_ocr_with_boxes(
        self,
        mixed_input: List[dict],
        bos_token_id: int,
        task: str = TaskNames.ocr_with_boxes,
    ):
        processed_input_ids = []
        all_image_tiles = []
        all_grid_thw = []

        # 1. Process the image input
        for i, input_dict in enumerate(mixed_input):
            processor_output = self._process_input(input_dict, task)
            input_ids = processor_output["input_ids"]
            image_tiles = processor_output["image_tiles"]
            grid_thw = processor_output["grid_thw"]

            # Special handling of some delimiter tokens
            if i == 1:
                assert input_dict["type"] == "text", (
                    "Expected text input for model input."
                )
                # Case for input - Add task specific bos token + end_of_input token
                # We do not want the model to learn how to predict inputs. Hence IGNORE_INDEX for these
                input_ids = [bos_token_id] + input_ids + [self.eoi_token_id]
            if i == 2:
                assert input_dict["type"] == "text", (
                    "Expected text for final model input"
                )
                input_ids = input_ids + [self.eos_token_id]
            elif i > 2:
                raise ValueError(f"Too many inputs received. Expected is 2 for inference, 3 for training. Received: {len(mixed_input)}")

            # Some input types don't return any image tiles, accounting for that
            if image_tiles is not None:
                all_image_tiles.append(image_tiles)
                all_grid_thw.append(grid_thw)

            processed_input_ids.extend(input_ids)

        return (
            torch.tensor(processed_input_ids, dtype=torch.long),
            all_image_tiles,
            all_grid_thw,
        )

    def _process_layout(self, mixed_input: List[dict], bos_token_id: int):
        return self._process_ocr_with_boxes(
            mixed_input, bos_token_id=bos_token_id, task="layout"
        )

    def _process_table_structure(self, mixed_input: List[dict], bos_token_id: int):
        return self._process_ocr_with_boxes(
            mixed_input, bos_token_id=bos_token_id, task="table_structure"
        )

    def _process_ocr_without_boxes(
        self,
        mixed_input: List[dict],
        bos_token_id: int,
        task: str = "ocr_without_boxes",
    ):
        # Boxes are set to None, so this will work
        # TODO: improve this behavior
        return self._process_ocr_with_boxes(
            mixed_input, bos_token_id=bos_token_id, task=task
        )

    def _process_block_without_boxes(
        self,
        mixed_input: List[dict],
        bos_token_id: int,
        task: str = "block_without_boxes",
    ):
        return self._process_ocr_with_boxes(
            mixed_input, bos_token_id=bos_token_id, task=task
        )

    def align_long_axis(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        height, width, _ = image.shape
        if height > width:  # Rotate vertical lines
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return image, True

        return image, False

    def __call__(
        self,
        mixed_batch: List[dict],
        padding_side: Optional[str] = "left",
        device: Optional[torch.device] = None,
        pad_to_multiple: Optional[int] = None,
    ):
        all_image_tiles = []
        all_input_ids = []
        all_grid_thw = []

        for b in mixed_batch:
            mixed_input = b["inputs"]
            task = b["task"]
            assert task in self.bos_token_id, f"Task {task} has no bos token defined."

            # Select the correct processing function based on the task type
            input_ids, image_tiles, grid_thw = getattr(self, f"_process_{task}")(
                mixed_input, self.bos_token_id[task]
            )

            all_input_ids.append(input_ids)
            all_image_tiles.extend(image_tiles)
            all_grid_thw.extend(grid_thw)

        batched_input_ids = pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_side=padding_side,
            padding_value=self.pad_token_id,
        )

        if pad_to_multiple is not None:
            current_len = batched_input_ids.shape[1]
            # Calculate the next multiple of pad_to_multiple
            padded_len = (
                (current_len + pad_to_multiple - 1) // pad_to_multiple
            ) * pad_to_multiple

            if padded_len > current_len:
                pad_len = padded_len - current_len
                batched_input_ids = torch.nn.functional.pad(
                    batched_input_ids, (pad_len, 0), value=self.pad_token_id
                )

        attention_mask = batched_input_ids.ne(self.pad_token_id)

        # Generating position IDs that are independent of left and right padding;
        # This should ensure same results for either padding side. Exact position id for the pad tokens themselves don't matter since they are masked
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids[position_ids < 0] = (
            0  # For left padding, the position ids for padding will become -1 because of the shift; Setting to 0
        )
        position_ids = (
            attention_mask.to(torch.long) * position_ids
        )  # Ensure right pad ids get set to zero

        batched_image_tiles = torch.cat(all_image_tiles, dim=0)
        batched_grid_thw = torch.from_numpy(np.array(all_grid_thw))

        # Pin memory for CUDA
        if device == torch.device("cuda"):
            batched_image_tiles = batched_image_tiles.pin_memory()
            batched_grid_thw = batched_grid_thw.pin_memory()
            attention_mask = attention_mask.pin_memory()
            batched_input_ids = batched_input_ids.pin_memory()
            position_ids = position_ids.pin_memory()

        return BatchFeature(
            {
                "input_ids": batched_input_ids,
                "image_tiles": batched_image_tiles,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "grid_thw": batched_grid_thw,
            }
        )

    # Decode model outputs; Strips special tokens
    def decode(self, tokens: List[int], task: str):
        filtered_tokens = [
            t
            for t in tokens
            if t not in self.special_token_mapping.values() and t != -100
        ]  # Skip special tokens and loss ignore index
        return self.ocr_tokenizer.decode(filtered_tokens, task=task)
