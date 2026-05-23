from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import torch
import math
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from surya.common.surya import SuryaModelOutput
from surya.common.xla import mark_step
from surya.common.predictor import BasePredictor

from surya.foundation.loader import FoundationModelLoader
from surya.foundation.util import (
    detect_repeat_token,
)
from surya.common.surya.schema import TaskNames
from surya.foundation.cache.dynamic_ops import DynamicOpsCache
from surya.foundation.cache.static_ops import StaticOpsCache

from surya.settings import settings
from surya.logging import get_logger, configure_logging

configure_logging()
logger = get_logger()


@dataclass
class ContinuousBatchInput:
    input_ids: torch.Tensor
    input_boxes: torch.Tensor
    position_ids: torch.Tensor
    # input_ids and position_ids may be padded, num_valid_tokens tracks the 'real' counts
    num_valid_tokens: torch.Tensor
    # count the number of predicted tokens for each batch element so far
    num_predicted_tokens: torch.Tensor
    needs_bbox_embedding: torch.Tensor


@dataclass
class ContinuousBatchOutput:
    input_ids: torch.Tensor
    preds: torch.Tensor
    bbox_preds: torch.Tensor
    scores: torch.Tensor
    token_probs: torch.Tensor


@dataclass
class FoundationPrompt:
    id: int
    task_name: TaskNames
    image: np.ndarray
    text: str
    math_mode: bool


class FoundationPredictor(BasePredictor):
    model_loader_cls = FoundationModelLoader
    batch_size = (
        settings.RECOGNITION_BATCH_SIZE
    )  # Default to the recognition batch size
    torch_dtype = None  # No default, loader picks the dtype based on device properties - bf16/fp16
    default_batch_sizes = {"cpu": 32, "mps": 64, "cuda": 256, "xla": 64}
    encoder_chunk_size: int = 4096  # Default chunk size
    encoder_chunk_sizes = {"cpu": 4096, "mps": 4096, "cuda": 32768, "xla": 32768}
    extra_token_count = {
        "xla": 128
    }  # We have to pad the XLA cache since we don't use sliding window
    min_prefill_ratio: int = 1 if settings.FOUNDATION_XLA else 0.2
    min_trim_length: int = 50
    tasks = {
        TaskNames.ocr_with_boxes: {
            "needs_bboxes": True,
            "img_size": (1024, 512),
            "max_tokens": 224,
        },
        TaskNames.ocr_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 512),
            "max_tokens": 224,
        },
        TaskNames.block_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 512),
            "max_tokens": 768,
        },
        TaskNames.layout: {
            "needs_bboxes": False,
            "img_size": (1024, 1024),
            "max_tokens": 200,
        },
        TaskNames.table_structure: {
            "needs_bboxes": False,
            "img_size": (1024, 512),
            "max_tokens": 600,
        },
    }

    def __init__(
        self,
        checkpoint=None,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=None,
        attention_implementation: Optional[str] = None,
    ):
        super().__init__(checkpoint, device, dtype, attention_implementation)
        self.prompt_queue = deque()
        self.batch_prompt_mapping = None
        self.kv_cache = None

        self.beacon_token_interval = self.model.config.beacon_token_interval

        # Setup various tokens on-device
        self.device_pad_token = torch.tensor(
            self.processor.pad_token_id, device=self.model.device, dtype=torch.long
        )
        self.device_beacon_token = torch.tensor(
            self.processor.beacon_token_id, device=self.model.device, dtype=torch.long
        )
        self.special_token_ids = torch.tensor(
            [self.model.config.image_token_id] + self.model.config.register_token_ids,
            device=self.model.device,
        )

        self.pad_to_multiple = (
            settings.FOUNDATION_PAD_TO_NEAREST
            if settings.FOUNDATION_STATIC_CACHE
            else None
        )

    def to(self, device_dtype: torch.device | str | None = None):
        super().to(device_dtype)
        self.special_token_ids = self.special_token_ids.to(device_dtype)

    def get_encoder_chunk_size(self) -> int:
        if settings.FOUNDATION_CHUNK_SIZE is not None:
            return settings.FOUNDATION_CHUNK_SIZE

        chunk_size = self.encoder_chunk_size
        if settings.TORCH_DEVICE_MODEL in self.encoder_chunk_sizes:
            if settings.TORCH_DEVICE_MODEL in self.encoder_chunk_sizes:
                chunk_size = self.encoder_chunk_sizes[settings.TORCH_DEVICE_MODEL]
        return chunk_size

    def setup_cache(self, batch_size: int, max_cache_len: int, max_sliding_window: int):
        kv_cache_cls = StaticOpsCache if settings.FOUNDATION_XLA else DynamicOpsCache
        self.kv_cache = kv_cache_cls(
            self.model.config,
            batch_size,
            max_cache_len,
            text_sliding_window=max_sliding_window,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        self.prompt_queue.clear()
        self.batch_prompt_mapping = {i: None for i in range(batch_size)}

    @property
    def num_empty_slots(self):
        return sum(v is None for v in self.batch_prompt_mapping.values())

    @property
    def num_active_slots(self):
        return len(self.batch_prompt_mapping) - self.num_empty_slots

    def prepare_input(
        self,
        task_names: List[str],
        images: List[Image.Image],
        input_text: List[str | None],
        math_modes: List[bool],
    ):
        batch = []
        for image, text, task_name, math_mode in zip(
            images, input_text, task_names, math_modes
        ):
            image_size = self.tasks[task_name]["img_size"]

            try:
                image = self.processor.scale_to_fit(
                    image, image_size
                )  # Only resizes if out of bounds (max/min)
            except cv2.error:
                # The image is empty if it can't be resized, so just make a blank image
                image = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)

            # Task input is the same for all tasks for now
            text = text or ""

            # Remove input text that exceeds max generation tokens (likely invalid)
            if len(text) > self.tasks[task_name]["max_tokens"]:
                text = ""
            inputs = [
                {"type": "image", "image": image, "rotated": False},
                {"type": "text", "text": text.strip(), "math": math_mode},
            ]
            batch.append({"task": task_name, "inputs": inputs})

        return batch

    def process_outputs(
        self, outputs: SuryaModelOutput, max_lookahead_tokens: Optional[int] = None
    ) -> ContinuousBatchOutput:
        # Predictions are multi-token
        lm_logits = outputs["lm_logits"].float()  # shape: [batch_size, seq_len, V]
        bbox_logits = outputs["bbox_logits"].float()  # shape: [batch_size, seq_len, 6]

        if (
            max_lookahead_tokens is not None
            and lm_logits.shape[1] > max_lookahead_tokens + 1
        ):
            lm_logits = lm_logits[:, : max_lookahead_tokens + 1, :]
            bbox_logits = bbox_logits[:, : max_lookahead_tokens + 1, :]

        # Get predictions
        preds = torch.argmax(lm_logits, dim=-1)
        input_ids = preds.to(torch.long)

        # Confidence scores for all tokens
        token_probs = F.softmax(lm_logits, dim=-1)
        scores = torch.max(token_probs, dim=-1).values  # shape: [B, T]

        # Update input boxes
        box_preds = bbox_logits * self.model.config.bbox_size
        box_preds = box_preds.to(torch.long)

        return ContinuousBatchOutput(
            input_ids=input_ids,
            preds=preds,
            bbox_preds=box_preds,
            scores=scores,
            token_probs=token_probs,
        )

    # Always left pad with beacons, don't worry about attention masking
    def maybe_insert_beacon_tokens(
        self,
        input_ids: torch.Tensor,
        input_boxes: torch.Tensor,
        num_predicted_tokens: torch.Tensor,
        num_new_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = (
            input_ids.shape
        )  # seq_len can be >1 - In case of multi-token predictions

        # num_predicted tokens **does not include** the current new input_ids, this number is updated **after beacon tokens are inserted**
        token_positions = num_predicted_tokens + torch.arange(
            1, seq_len + 1, device=input_ids.device
        ).unsqueeze(0)
        beacon_positions = token_positions % self.beacon_token_interval == 0

        # If no beacons needed, return original input
        needs_beacon = beacon_positions.any(dim=1)  # shape: [batch_size]
        if not needs_beacon.any():
            if num_new_tokens is None:
                num_new_tokens = (
                    torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
                    * seq_len
                )
            return input_ids, input_boxes, num_new_tokens.squeeze(1)

        beacon_insert_pos = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        for i in range(batch_size):
            if needs_beacon[i]:
                # Find first position that needs beacon
                beacon_insert_pos[i] = torch.where(beacon_positions[i])[0]

        # Padded input ids.
        new_input_ids = torch.full(
            (batch_size, seq_len + 1),
            self.device_pad_token,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        new_input_boxes = torch.full(
            (batch_size, seq_len + 1, 6),
            -100,
            dtype=input_boxes.dtype,
            device=input_boxes.device,
        )
        # Fill in tokens for each sequence
        for i in range(batch_size):
            if needs_beacon[i]:
                insert_pos = beacon_insert_pos[i]
                new_input_ids[i, insert_pos] = self.device_beacon_token
                new_input_boxes[i, insert_pos, :] = -100
                if insert_pos > 0:
                    new_input_ids[i, :insert_pos] = input_ids[i, :insert_pos]
                    new_input_boxes[i, :insert_pos] = input_boxes[i, :insert_pos]
                new_input_ids[i, insert_pos + 1 :] = input_ids[i, insert_pos:]
                new_input_boxes[i, insert_pos + 1 :] = input_boxes[i, insert_pos:]
            else:
                new_input_ids[i, 1:] = input_ids[i, :]
                new_input_boxes[i, 1:] = input_boxes[i, :]

        # Calculate valid token counts for both padded and non padded sequences
        valid_token_counts = torch.where(
            needs_beacon,
            torch.tensor(seq_len + 1, device=input_ids.device),
            torch.tensor(seq_len, device=input_ids.device),
        )

        return new_input_ids, new_input_boxes, valid_token_counts

    def decode(
        self,
        current_inputs: Optional[ContinuousBatchInput] = None,
        max_lookahead_tokens: Optional[int] = None,
    ):
        # Note - If we want to use the outputs from the non-last token, we
        # need to set the cache position manually to ensure causality. The default
        # behavior only works for the last token currently
        input_ids = current_inputs.input_ids
        input_boxes = current_inputs.input_boxes
        embed_boxes = current_inputs.needs_bbox_embedding

        position_ids = current_inputs.position_ids
        num_predicted_tokens = current_inputs.num_predicted_tokens
        num_valid_tokens = current_inputs.num_valid_tokens
        batch_size = input_ids.shape[0]

        # Pre-shift the attention mask based on the cache update
        self.kv_cache.decode_attention_mask_update(
            num_valid_tokens=num_valid_tokens, cache_idxs=list(range(batch_size))
        )

        cache_position = self.get_cache_position(
            input_ids.shape[1], self.kv_cache.attention_mask, prefill=False
        )
        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=self.kv_cache.attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                past_key_values=self.kv_cache,
                prefill=False,
                num_valid_tokens=num_valid_tokens,
                input_boxes=input_boxes,
                embed_boxes=embed_boxes,
                logits_to_keep=1,
            )

        processed_output: ContinuousBatchOutput = self.process_outputs(
            outputs, max_lookahead_tokens=max_lookahead_tokens
        )

        input_ids = processed_output.input_ids
        input_boxes = processed_output.bbox_preds

        # Update this **before** inserting beacon tokens
        tau = settings.FOUNDATION_MULTI_TOKEN_MIN_CONFIDENCE
        if max_lookahead_tokens is not None:
            num_new_tokens = torch.clamp(
                (
                    processed_output.scores.ge(tau)
                    .to(torch.long)
                    .cumprod(dim=1)
                    .sum(dim=1, keepdim=True)
                ),
                min=1,
            )
        else:
            num_new_tokens = input_ids.shape[1]

        num_predicted_tokens += num_new_tokens
        input_ids, input_boxes, num_valid_tokens = self.maybe_insert_beacon_tokens(
             input_ids, input_boxes, num_predicted_tokens, num_new_tokens
        )
        position_ids = position_ids[:, -1:] + torch.arange(
            1, input_ids.shape[1] + 1, device=input_ids.device
        )
        # Some of the input sequences may now have left padding tokens, so we want to account for that
        # offset is a per-batch offset of the position_ids
        offset = (input_ids.shape[1] - num_valid_tokens).unsqueeze(1)
        position_ids -= offset

        new_input = ContinuousBatchInput(
            input_ids=input_ids,
            input_boxes=input_boxes,
            position_ids=position_ids,
            num_valid_tokens=num_valid_tokens,
            num_predicted_tokens=num_predicted_tokens,
            needs_bbox_embedding=current_inputs.needs_bbox_embedding,
        )

        return new_input, processed_output

    def pad_and_shift_input_ids_position_ids(
        self,
        input_ids: torch.Tensor,
        bbox_preds: torch.Tensor,
        position_ids: torch.Tensor,
        new_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pads new_input_ids to match the new seq len with **left padding**
        and creates updated position_ids

        Returns:
            padded_input_ids (torch.Tensor): [batch_size, current_seq_len]
            updated_position_ids (torch.Tensor): [batch_size, current_seq_len]
        """
        # No padding
        if new_seq_len == input_ids.shape[1]:
            return (
                input_ids,
                bbox_preds,
                position_ids[:, -1:] + torch.arange(1, new_seq_len + 1, device=self.model.device),
            )

        pad_len = new_seq_len - input_ids.shape[1]
        padded_input_ids = torch.nn.functional.pad(
            input_ids, (pad_len, 0), value=self.device_pad_token
        )

        padded_bbox_preds = torch.nn.functional.pad(
            bbox_preds, (0, 0, pad_len, 0), value=-100
        )

        # Since we have **left padding**, offset the new position_ids by the amount of padding
        # This ensures that the **true tokens** get the correct position_ids
        # The position_ids assigned to pad tokens do not matter. They are not cached, and not used for outputs
        updated_position_ids = position_ids[:, -1:] + torch.arange(
            1, new_seq_len + 1, device=self.model.device
        )
        updated_position_ids -= pad_len

        return padded_input_ids, padded_bbox_preds, updated_position_ids

    def get_cache_position(
        self,
        seq_len: int,
        attention_mask: torch.Tensor,
        prefill: bool,
    ):
        batch_size, target_len = attention_mask.shape
        base_cache_position = (
            torch.arange(seq_len, device=attention_mask.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        if prefill:
            return base_cache_position

        # This is a (batch_size) tensor, we can add the seq lens here
        cache_seqlens = (
            attention_mask
            * torch.arange(attention_mask.size(1), device=attention_mask.device)
        ).argmax(dim=1).to(torch.int32) + 1
        # Needs to be unsqueezed so broadcasting works
        return cache_seqlens.unsqueeze(1) + base_cache_position

    def prefill(
        self,
        current_inputs: Optional[ContinuousBatchInput] = None,
        max_lookahead_tokens: Optional[int] = None,
    ):
        logger.debug(f"Prefilling {self.num_empty_slots} slots")

        prompts: List[FoundationPrompt] = [
            self.prompt_queue.popleft()
            for _ in range(min(self.num_empty_slots, len(self.prompt_queue)))
        ]
        non_active_idxs = [k for k, v in self.batch_prompt_mapping.items() if v is None]
        idxs_to_merge = non_active_idxs[: len(prompts)]

        for i, prompt in zip(idxs_to_merge, prompts):
            self.batch_prompt_mapping[i] = prompt.id

        needs_bbox_embedding = torch.tensor(
            [
                p.task_name in [TaskNames.layout, TaskNames.table_structure]
                for p in prompts
            ],
            dtype=torch.bool,
        )

        batch_input = self.prepare_input(
            task_names=[p.task_name for p in prompts],
            images=[p.image for p in prompts],
            input_text=[p.text for p in prompts],
            math_modes=[
                p.math_mode for p in prompts
            ],  # Pass math mode to the processor
        )
        processed_inputs = self.processor(
            batch_input,
            padding_side="left",
            device=self.model.device,
            pad_to_multiple=self.pad_to_multiple,
        )

        input_ids = processed_inputs["input_ids"].to(dtype=torch.long)
        attention_mask = processed_inputs["attention_mask"].to(dtype=torch.long)
        position_ids = processed_inputs["position_ids"].to(dtype=torch.long)
        valid_batch_size = len(idxs_to_merge)

        # Keep these off device until later
        image_tiles = processed_inputs["image_tiles"].to(dtype=self.model.dtype)
        grid_thw = processed_inputs["grid_thw"].to(dtype=torch.long)

        if settings.FOUNDATION_STATIC_CACHE:
            input_ids = self.pad_to_batch_size(
                input_ids, batch_size=self.kv_cache.max_batch_size
            )
            attention_mask = self.pad_to_batch_size(
                attention_mask, batch_size=self.kv_cache.max_batch_size
            )
            position_ids = self.pad_to_batch_size(
                position_ids, batch_size=self.kv_cache.max_batch_size
            )
            needs_bbox_embedding = self.pad_to_batch_size(
                needs_bbox_embedding, batch_size=self.kv_cache.max_batch_size
            )

        # Move to device after padding
        input_ids = input_ids.to(device=self.model.device)
        attention_mask = attention_mask.to(device=self.model.device)
        position_ids = position_ids.to(device=self.model.device)
        needs_bbox_embedding = needs_bbox_embedding.to(device=self.model.device)

        # Find text lengths of each
        # Oddly, this is optimal on GPU - causes a 30% slowdown if "optimized"
        # Be very careful with the type and device of this - can cause
        # a big slowdown if put on device
        is_special = (
            (input_ids.unsqueeze(-1) == self.special_token_ids).any(-1).cpu()
        )  # (batch, seq_len)
        text_lengths = []
        for i in range(input_ids.shape[0]):
            special_positions = is_special[i].nonzero(as_tuple=True)[0]
            if len(special_positions) > 0:
                # Assuming special tokens are contiguous at the start
                prefix_len = special_positions[-1].item() + 1
            else:
                prefix_len = 0
            text_lengths.append(input_ids.shape[1] - prefix_len)
        text_lengths = torch.tensor(text_lengths, dtype=torch.long)

        cache_position = self.get_cache_position(
            input_ids.shape[1], attention_mask, prefill=True
        )
        with settings.INFERENCE_MODE():
            image_embeddings = self.model.get_image_embeddings(
                pixel_values=image_tiles,
                grid_thw=grid_thw,
                encoder_chunk_size=self.get_encoder_chunk_size(),
                valid_batch_size=valid_batch_size,
                max_batch_size=self.kv_cache.max_batch_size,
            )
            mark_step()

            outputs = self.model(
                input_ids=input_ids,
                image_embeddings=image_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                inputs_embeds=None,
                past_key_values=self.kv_cache,
                use_cache=True,
                encoder_chunk_size=self.get_encoder_chunk_size(),
                cache_idxs=idxs_to_merge,
                prefill=True,
                num_valid_tokens=None,  # Not required during prefill
                text_lengths=text_lengths,
                valid_batch_size=valid_batch_size,
                logits_to_keep=1,
            )

        # Process outputs
        processed_outputs = self.process_outputs(
            outputs, max_lookahead_tokens=max_lookahead_tokens
        )
        # Multi-token prediction
        predicted_tokens = processed_outputs.input_ids.shape[1]
        num_valid_tokens = (
            torch.ones((input_ids.shape[0]), device=self.model.device, dtype=torch.long)
            * predicted_tokens
        )
        num_predicted_tokens = (
            torch.ones(
                (input_ids.shape[0], 1), device=self.model.device, dtype=torch.long
            )
            * predicted_tokens
        )

        self.kv_cache.prefill_attention_mask_update(
            attention_mask, idxs_to_merge, valid_batch_size, text_lengths
        )
        self.kv_cache.update_text_counts(idxs_to_merge, valid_batch_size, text_lengths)

        full_batch = len(idxs_to_merge) == self.kv_cache.max_batch_size

        # If full batch, then we can ignore current_inputs
        if current_inputs is None or full_batch:
            new_seq_len = processed_outputs.input_ids.shape[1]
            # No padding tokens - So we can safely set position_ids this way
            position_ids = position_ids[:, -1:] + torch.arange(
                1, new_seq_len + 1, device=position_ids.device
            )
            new_input = ContinuousBatchInput(
                input_ids=processed_outputs.input_ids,
                input_boxes=processed_outputs.bbox_preds,
                position_ids=position_ids,
                num_valid_tokens=num_valid_tokens,
                num_predicted_tokens=num_predicted_tokens,
                needs_bbox_embedding=needs_bbox_embedding,
            )

            return (
                new_input,
                processed_outputs,
                range(processed_outputs.input_ids.shape[0]),
            )

        # Merging inputs for next steps
        current_input_ids = current_inputs.input_ids
        current_position_ids = current_inputs.position_ids
        current_input_boxes = current_inputs.input_boxes

        current_needs_bbox_embedding = current_inputs.needs_bbox_embedding

        assert current_input_ids.shape[1] == current_position_ids.shape[1]
        input_ids, bbox_preds, position_ids = self.pad_and_shift_input_ids_position_ids(
            processed_outputs.input_ids,
            processed_outputs.bbox_preds,
            position_ids,
            new_seq_len=current_input_ids.shape[1],
        )

        current_input_ids[idxs_to_merge] = input_ids[:valid_batch_size]
        current_input_boxes[idxs_to_merge] = bbox_preds[:valid_batch_size]
        current_position_ids[idxs_to_merge] = position_ids[:valid_batch_size]

        current_num_valid_tokens = current_inputs.num_valid_tokens
        current_num_valid_tokens[idxs_to_merge] = num_valid_tokens[:valid_batch_size]

        current_num_predicted_tokens = current_inputs.num_predicted_tokens
        current_num_predicted_tokens[idxs_to_merge] = num_predicted_tokens[
            :valid_batch_size
        ]
        current_needs_bbox_embedding[idxs_to_merge] = needs_bbox_embedding[
            :valid_batch_size
        ]

        new_input = ContinuousBatchInput(
            input_ids=current_input_ids,
            input_boxes=current_input_boxes,
            position_ids=current_position_ids,
            num_valid_tokens=current_num_valid_tokens,
            num_predicted_tokens=current_num_predicted_tokens,
            needs_bbox_embedding=current_needs_bbox_embedding,
        )

        return new_input, processed_outputs, idxs_to_merge

    def get_max_image_token_count(
        self, images: list[np.ndarray], tasks: List[TaskNames]
    ) -> int:
        def compute_scaled_size(
            H: int, W: int, max_size: Tuple[int, int]
        ) -> Tuple[int, int]:
            max_W, max_H = max_size
            min_W, min_H = (168, 168)

            current_pixels = H * W
            max_pixels = max_H * max_W
            min_pixels = min_H * min_W
            current_pixels = max(1, current_pixels)  # Avoid zero division

            if current_pixels > max_pixels:
                scale = (max_pixels / current_pixels) ** 0.5
                return math.floor(H * scale), math.floor(W * scale)
            elif current_pixels < min_pixels:
                scale = (min_pixels / current_pixels) ** 0.5
                return math.ceil(H * scale), math.ceil(W * scale)
            return H, W

        def get_tile_count(H: int, W: int, factor: int) -> int:
            H_bar = math.ceil(H / factor) * factor
            W_bar = math.ceil(W / factor) * factor
            grid_h = H_bar / self.processor.patch_size
            grid_w = W_bar // self.processor.patch_size
            return grid_h * grid_w

        max_tokens = 0
        factor = self.processor.patch_size * self.processor.merge_size

        for image, task in zip(images, tasks):
            H, W = image.shape[:2]
            max_size = self.tasks[task]["img_size"]
            scaled_H, scaled_W = compute_scaled_size(H, W, max_size)
            token_count = get_tile_count(scaled_H, scaled_W, factor) / (
                self.processor.merge_size**2
            )
            max_tokens = max(max_tokens, token_count)

        # Extra 10 to account for EOS/BOS/Rotation token etc.
        return 10 + self.processor.num_register_tokens + int(max_tokens)

    def prediction_loop(
        self,
        images: List[np.ndarray],
        input_texts: List[str],
        task_names: List[TaskNames],
        batch_size: int | None = None,
        max_tokens: int | None = None,
        max_sliding_window: int | None = None,
        math_mode: bool = True,
        drop_repeated_tokens: bool = True,
        max_lookahead_tokens: Optional[int] = None,
        top_k: int = 0,
        tqdm_desc: str = "Recognizing Text"
    ) -> tuple:
        allowed_tasks = self.tasks.keys()
        assert all([task_name in allowed_tasks for task_name in task_names]), (
            f"One or more tasks in {task_names} is not supported. Supported tasks are {allowed_tasks}"
        )

        predicted_tokens = [[] for _ in range(len(images))]
        scores = [[] for _ in range(len(images))]
        topk_probs = [[] for _ in range(len(images))]

        if batch_size is None:
            batch_size = self.get_batch_size()

        batch_size = min(len(images), batch_size)
        current_inputs = None

        max_image_tokens = self.get_max_image_token_count(images, task_names)
        if max_sliding_window is None:
            max_sliding_window = self.model.config.sliding_window
        self.setup_cache(
            batch_size,
            max_cache_len=max_image_tokens + max_sliding_window + self.extra_token_count.get(settings.TORCH_DEVICE_MODEL, 0),
            max_sliding_window=max_sliding_window,
        )

        batch_max_tokens = {}
        for idx, (img, txt, task) in enumerate(zip(images, input_texts, task_names)):
            self.prompt_queue.append(
                FoundationPrompt(
                    id=idx, task_name=task, text=txt, image=img, math_mode=math_mode
                )
            )
            batch_max_tokens[idx] = (
                max_tokens
                or settings.FOUNDATION_MAX_TOKENS
                or self.tasks[task]["max_tokens"]
            )

        overall_max_tokens = max(batch_max_tokens.values())

        pbar = tqdm(
            total=len(self.prompt_queue),
            desc=tqdm_desc,
            disable=self.disable_tqdm,
        )

        batch_bboxes = torch.zeros(len(images), overall_max_tokens, 6)
        batch_pos = [0] * len(images)

        while self.prompt_queue or self.num_active_slots > 0:
            if (
                self.num_empty_slots / batch_size
            ) >= self.min_prefill_ratio and self.prompt_queue:
                updated_inputs, outputs, merge_idxs = self.prefill(
                    current_inputs, max_lookahead_tokens=0
                )

                predicted_tokens_cpu = outputs.preds.cpu()
                scores_cpu = outputs.scores.cpu()
                bbox_preds_cpu = outputs.bbox_preds.cpu()

                if top_k > 0:
                    batch_top_k_probs, batch_top_k_indices = torch.topk(
                        outputs.token_probs, k=top_k, dim=-1
                    )
                    batch_top_k_probs_cpu = batch_top_k_probs.cpu()
                    batch_top_k_indices_cpu = batch_top_k_indices.cpu()

                for temp_idx, b_idx in enumerate(merge_idxs):
                    if self.batch_prompt_mapping[b_idx] is not None:
                        p_idx = self.batch_prompt_mapping[b_idx]
                        seq_len = predicted_tokens_cpu.shape[1]
                        for t_idx in range(seq_len):
                            token = predicted_tokens_cpu[temp_idx, t_idx].item()
                            predicted_tokens[p_idx].append(token)
                            batch_bboxes[p_idx, batch_pos[p_idx]] = bbox_preds_cpu[
                                temp_idx, t_idx
                            ]
                            batch_pos[p_idx] += 1
                            scores[p_idx].append(scores_cpu[temp_idx, t_idx].item())

                            if top_k > 0:
                                top_k_scores = {
                                    batch_top_k_indices_cpu[temp_idx, t_idx][
                                        k
                                    ].item(): batch_top_k_probs_cpu[temp_idx, t_idx][
                                        k
                                    ].item()
                                    for k in range(top_k)
                                }
                                topk_probs[p_idx].append(top_k_scores)

                            if token in [
                                self.processor.eos_token_id,
                                self.processor.no_output_token,
                            ]:
                                self.batch_prompt_mapping[b_idx] = None
                                pbar.update(1)
                                break
            else:
                updated_inputs, outputs = self.decode(
                    current_inputs, max_lookahead_tokens=max_lookahead_tokens
                )
                mark_step()

                predicted_tokens_cpu = outputs.preds.cpu()
                scores_cpu = outputs.scores.cpu()
                bbox_preds_cpu = outputs.bbox_preds.cpu()

                if top_k > 0:
                    batch_top_k_probs, batch_top_k_indices = torch.topk(
                        outputs.token_probs, k=top_k, dim=-1
                    )
                    batch_top_k_probs_cpu = batch_top_k_probs.cpu()
                    batch_top_k_indices_cpu = batch_top_k_indices.cpu()

                for b_idx, p_idx in self.batch_prompt_mapping.items():
                    if p_idx is not None:
                        seq_len = predicted_tokens_cpu.shape[1]
                        num_tokens = updated_inputs.num_valid_tokens[b_idx].item()
                        should_stop = False

                        for t_idx in range(seq_len):
                            # don't use multitoken prediction for lower confidence tokens
                            if t_idx > 0 and num_tokens < seq_len:
                                # roll so tokens are right aligned
                                updated_inputs.input_ids[b_idx] = (
                                    updated_inputs.input_ids[b_idx].roll(
                                        shifts=seq_len - num_tokens, dims=0
                                    )
                                )
                                # don't need to roll position_ids because that's handled in `decode` (and when we do beacon tokens)
                                break

                            token = predicted_tokens_cpu[b_idx, t_idx].item()
                            predicted_tokens[p_idx].append(token)
                            batch_bboxes[p_idx, batch_pos[p_idx]] = bbox_preds_cpu[
                                b_idx, t_idx
                            ]
                            batch_pos[p_idx] += 1
                            scores[p_idx].append(scores_cpu[b_idx, t_idx].item())

                            if top_k > 0:
                                top_k_scores = {
                                    batch_top_k_indices_cpu[temp_idx, t_idx][
                                        k
                                    ].item(): batch_top_k_probs_cpu[temp_idx, t_idx][
                                        k
                                    ].item()
                                    for k in range(top_k)
                                }
                                topk_probs[p_idx].append(top_k_scores)

                            repeats = len(predicted_tokens[p_idx]) >= batch_max_tokens[
                                p_idx
                            ] or (
                                drop_repeated_tokens
                                and detect_repeat_token(predicted_tokens[p_idx])
                                and task_names[p_idx]
                                in [
                                    TaskNames.ocr_with_boxes,
                                    TaskNames.ocr_without_boxes,
                                ]
                            )
                            if (
                                token
                                in [
                                    self.processor.eos_token_id,
                                    self.processor.pad_token_id,
                                ]
                                or repeats
                            ):
                                should_stop = True
                                break

                        if should_stop:
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)

            # Update inputs and mark XLA step
            current_inputs = updated_inputs

        pbar.close()

        del self.kv_cache
        self.kv_cache = None
        torch.cuda.empty_cache()

        return predicted_tokens, batch_bboxes, scores, topk_probs
