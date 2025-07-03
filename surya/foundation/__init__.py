from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import QuantizedCacheConfig

from surya.common.surya import SuryaModelOutput
from surya.common.util import mark_step
from surya.common.predictor import BasePredictor

from surya.foundation.loader import FoundationModelLoader
from surya.foundation.util import (
    detect_repeat_token,
)
from surya.common.surya.schema import TaskNames
from surya.foundation.cache import (
    ContinuousBatchingCache,
)
from surya.settings import settings
from surya.logging import get_logger, configure_logging

configure_logging()
logger = get_logger()


@dataclass
class ContinuousBatchInput:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    # input_ids and position_ids may be padded, num_valid_tokens tracks the 'real' counts
    num_valid_tokens: torch.Tensor
    # count the number of predicted tokens for each batch element so far
    num_predicted_tokens: torch.Tensor


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
    batch_size = settings.RECOGNITION_BATCH_SIZE        # Default to the recognition batch size
    torch_dtype = None  # No default, loader picks the dtype based on device properties - bf16/fp16
    default_batch_sizes = {"cpu": 32, "mps": 64, "cuda": 256, "xla": 128}
    encoder_chunk_size: int = 4096  # Default chunk size
    encoder_chunk_sizes = {"cpu": 4096, "mps": 4096, "cuda": 32768, "xla": 32768}
    min_prefill_ratio: int = 0.2
    min_trim_length: int = 50
    tasks = {
        TaskNames.ocr_with_boxes: {
            "needs_bboxes": True,
            "img_size": (1024, 256),  # 370 max tokens
            "max_tokens": 224,
        },
        TaskNames.ocr_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 256),  # 370 max tokens
            "max_tokens": 224,
        },
        TaskNames.block_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 512),  # 703 max tokens
            "max_tokens": 768,
        },
        TaskNames.layout: {
            "needs_bboxes": False,
            "img_size": (1024, 1024),
            "max_tokens": 200,
        },
    }

    def __init__(self, checkpoint=None, device=settings.TORCH_DEVICE_MODEL, dtype=None):
        super().__init__(checkpoint, device, dtype)
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

        self.pad_to_multiple = 512 if settings.FOUNDATION_STATIC_CACHE else None

    def get_encoder_chunk_size(self) -> int:
        if settings.FOUNDATION_CHUNK_SIZE is not None:
            return settings.FOUNDATION_CHUNK_SIZE

        chunk_size = self.encoder_chunk_size
        if settings.TORCH_DEVICE_MODEL in self.encoder_chunk_sizes:
            if settings.TORCH_DEVICE_MODEL in self.encoder_chunk_sizes:
                chunk_size = self.encoder_chunk_sizes[settings.TORCH_DEVICE_MODEL]
        return chunk_size

    def setup_cache(self, batch_size: int, max_cache_len: int):
        self.kv_cache = ContinuousBatchingCache(
            self.model.config,
            batch_size,
            max_cache_len,
            text_sliding_window=self.model.config.sliding_window,
            device=self.model.device,
            dtype=self.model.dtype
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

    def process_outputs(self, outputs: SuryaModelOutput, max_lookahead_tokens: Optional[int]=None) -> ContinuousBatchOutput:
        # Predictions are multi-token
        lm_logits = outputs["lm_logits"].float()  # shape: [batch_size, seq_len, V]
        bbox_logits = outputs["bbox_logits"].float()  # shape: [batch_size, seq_len, 6]

        if max_lookahead_tokens is not None and lm_logits.shape[1] > max_lookahead_tokens + 1:
            lm_logits = lm_logits[:, :max_lookahead_tokens + 1, :]
            bbox_logits = bbox_logits[:, :max_lookahead_tokens + 1, :]
        
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
            token_probs=token_probs
        )

    # Make space for beacon tokens to be inserted while keeping the same seq len across all batch elements
    # This involves **left padding** the input sequence. Although this is unconventional, it works better
    # with the causal mask of flash attention, and we are careful to ignore this pad token when inserting
    # into cache
    def maybe_insert_beacon_tokens(
        self,
        input_ids: torch.Tensor,
        num_predicted_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape       # seq_len can be >1 - In case of multi-token predictions
        
        # num_predicted tokens **does not include** the current new input_ids, this number is updated **after beacon tokens are inserted**
        token_positions = num_predicted_tokens + torch.arange(1, seq_len + 1, device=input_ids.device).unsqueeze(0)
        beacon_positions = (token_positions % self.beacon_token_interval == 0)

        # If no beacons needed, return original input
        needs_beacon = beacon_positions.any(dim=1)  # shape: [batch_size]
        if not needs_beacon.any():
            return input_ids, torch.ones(batch_size, dtype=torch.long, device=input_ids.device) * seq_len
        
        beacon_insert_pos = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        for i in range(batch_size):
            if needs_beacon[i]:
                # Find first position that needs beacon
                beacon_insert_pos[i] = torch.where(beacon_positions[i])[0]
      
        # Padded input ids.
        new_input_ids = torch.full((batch_size, seq_len + 1), self.device_pad_token, 
                                dtype=input_ids.dtype, device=input_ids.device)
        
        # Fill in tokens for each sequence
        for i in range(batch_size):
            if needs_beacon[i]:
                insert_pos = beacon_insert_pos[i]
                new_input_ids[i, insert_pos] = self.device_beacon_token
                if insert_pos > 0:
                    new_input_ids[i, :insert_pos] = input_ids[i, :insert_pos]
                new_input_ids[i, insert_pos+1:] = input_ids[i, insert_pos:]
            else:
                new_input_ids[i, 1:] = input_ids[i, :]
        
        # Calculate valid token counts for both padded and non padded sequences
        valid_token_counts = torch.where(
            needs_beacon, 
            torch.tensor(seq_len + 1, device=input_ids.device),
            torch.tensor(seq_len, device=input_ids.device)
        )
        
        return new_input_ids, valid_token_counts

    def decode(self, current_inputs: Optional[ContinuousBatchInput] = None, max_lookahead_tokens: Optional[int] = None):
        # Note - If we want to use the outputs from the non-last token, we 
        # need to set the cache position manually to ensure causality. The default
        # behavior only works for the last token currently
        input_ids = current_inputs.input_ids
        position_ids = current_inputs.position_ids
        num_predicted_tokens = current_inputs.num_predicted_tokens
        num_valid_tokens = current_inputs.num_valid_tokens
        batch_size = input_ids.shape[0]

        # Pre-shift the attention mask based on the cache update
        self.kv_cache.maybe_shift_attention_mask(
            num_valid_tokens=num_valid_tokens, cache_idxs=list(range(batch_size))
        )
        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=self.kv_cache.attention_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=self.kv_cache,
                prefill=False,
                num_valid_tokens=num_valid_tokens
            )

        processed_output: ContinuousBatchOutput = self.process_outputs(outputs, max_lookahead_tokens=max_lookahead_tokens)
        
        input_ids = processed_output.input_ids

        # Update this **before** inserting beacon tokens
        num_new_tokens = input_ids.shape[1]
        num_predicted_tokens += num_new_tokens

        input_ids, num_valid_tokens = self.maybe_insert_beacon_tokens(input_ids, num_predicted_tokens)
        position_ids = position_ids[:, -1:] + torch.arange(1, input_ids.shape[1] + 1, device=input_ids.device)
        # Some of the input sequences may now have left padding tokens, so we want to account for that
        # offset is a per-batch offset of the position_ids
        offset = (input_ids.shape[1] - num_valid_tokens).unsqueeze(1)
        position_ids -= offset

        new_input = ContinuousBatchInput(
            input_ids=input_ids,
            position_ids=position_ids,
            num_valid_tokens=num_valid_tokens,
            num_predicted_tokens=num_predicted_tokens
        )

        return new_input, processed_output

    def pad_and_shift_input_ids_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        new_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads new_input_ids to match the new seq len with **left padding**
        and creates updated position_ids

        Returns:
            padded_input_ids (torch.Tensor): [batch_size, current_seq_len]
            updated_position_ids (torch.Tensor): [batch_size, current_seq_len]
        """
        # No padding
        if new_seq_len == input_ids.shape[1]:
            return input_ids, position_ids[:, -1:] + torch.arange(1, new_seq_len +1, device=self.model.device)

        pad_len = new_seq_len - input_ids.shape[1]
        padded_input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=self.device_pad_token)

        # Since we have **left padding**, offset the new position_ids by the amount of padding
        # This ensures that the **true tokens** get the correct position_ids
        # The position_ids assigned to pad tokens do not matter. They are not cached, and not used for outputs
        updated_position_ids = position_ids[:, -1:] + torch.arange(1, new_seq_len + 1, device=self.model.device)
        updated_position_ids -= pad_len

        return padded_input_ids, updated_position_ids

    def prefill(self, current_inputs: Optional[ContinuousBatchInput] = None, max_lookahead_tokens: Optional[int] = None):
        logger.debug(f"Prefilling {self.num_empty_slots} slots")
        prompts: List[FoundationPrompt] = [
            self.prompt_queue.popleft()
            for _ in range(min(self.num_empty_slots, len(self.prompt_queue)))
        ]
        non_active_idxs = [k for k, v in self.batch_prompt_mapping.items() if v is None]
        idxs_to_merge = non_active_idxs[: len(prompts)]
        for i, prompt in zip(idxs_to_merge, prompts):
            self.batch_prompt_mapping[i] = prompt.id

        batch_input = self.prepare_input(
            task_names=[p.task_name for p in prompts],
            images=[p.image for p in prompts],
            input_text=[p.text for p in prompts],
            math_modes=[
                p.math_mode for p in prompts
            ],  # Pass math mode to the processor
        )
        processed_inputs = self.processor(
            batch_input, padding_side="left", device=self.model.device, pad_to_multiple=self.pad_to_multiple
        ).to(device=self.model.device)

        input_ids = processed_inputs["input_ids"].to(dtype=torch.long)
        image_tiles = processed_inputs["image_tiles"].to(dtype=self.model.dtype)
        grid_thw = processed_inputs["grid_thw"].to(dtype=torch.long)
        attention_mask = processed_inputs["attention_mask"].to(dtype=torch.long)
        position_ids = processed_inputs["position_ids"].to(dtype=torch.long)
        valid_batch_size = len(idxs_to_merge)

        if settings.FOUNDATION_STATIC_CACHE:
            input_ids = self.pad_to_batch_size(input_ids, batch_size=self.kv_cache.max_batch_size)
            attention_mask = self.pad_to_batch_size(attention_mask, batch_size=self.kv_cache.max_batch_size)
            position_ids = self.pad_to_batch_size(position_ids, batch_size=self.kv_cache.max_batch_size)

        # Find text lengths of each
        is_special = (input_ids.unsqueeze(-1) == self.special_token_ids).any(-1)  # (batch, seq_len)
        text_lengths = []
        for i in range(input_ids.shape[0]):
            special_positions = is_special[i].nonzero(as_tuple=True)[0]
            if len(special_positions) > 0:
                # Assuming special tokens are contiguous at the start
                prefix_len = special_positions[-1].item() + 1
            else:
                prefix_len = 0
            text_lengths.append(input_ids.shape[1] - prefix_len)

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                image_tiles=image_tiles,
                grid_thw=grid_thw,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                past_key_values=self.kv_cache,
                use_cache=True,
                encoder_chunk_size=self.get_encoder_chunk_size(),
                cache_idxs=idxs_to_merge,
                prefill=True,
                num_valid_tokens=None,   # Not required during prefill
                text_lengths=text_lengths,
            )
        
        # Process outputs
        processed_outputs = self.process_outputs(outputs, max_lookahead_tokens=max_lookahead_tokens)
        # Multi-token prediction
        predicted_tokens = processed_outputs.input_ids.shape[1]
        num_valid_tokens = torch.ones((input_ids.shape[0]), device=self.model.device, dtype=torch.long) * predicted_tokens
        num_predicted_tokens = torch.ones((input_ids.shape[0], 1), device=self.model.device, dtype=torch.long) * predicted_tokens

        self.kv_cache.prefill_attention_mask_update(attention_mask, idxs_to_merge, text_lengths[:valid_batch_size])
        self.kv_cache.update_text_counts(idxs_to_merge, text_lengths[:valid_batch_size])

        if current_inputs is None:
            new_seq_len = processed_outputs.input_ids.shape[1]
            # No padding tokens - So we can safely set position_ids this way
            position_ids = position_ids[:, -1:] + torch.arange(1, new_seq_len + 1, device=position_ids.device)
            new_input = ContinuousBatchInput(
                input_ids=processed_outputs.input_ids,
                position_ids=position_ids,
                num_valid_tokens=num_valid_tokens,
                num_predicted_tokens=num_predicted_tokens,
            )

            return (
                new_input,
                processed_outputs,
                range(processed_outputs.input_ids.shape[0]),
            )

        # Merging inputs for next steps
        current_input_ids = current_inputs.input_ids
        current_position_ids = current_inputs.position_ids

        assert(current_input_ids.shape[1] == current_position_ids.shape[1])
        input_ids, position_ids = self.pad_and_shift_input_ids_position_ids(
            processed_outputs.input_ids, position_ids, new_seq_len=current_input_ids.shape[1]
        )
        current_input_ids[idxs_to_merge] = input_ids[:valid_batch_size]
        current_position_ids[idxs_to_merge] = position_ids[:valid_batch_size]

        current_num_valid_tokens = current_inputs.num_valid_tokens
        current_num_valid_tokens[idxs_to_merge] = num_valid_tokens[:valid_batch_size]

        current_num_predicted_tokens = current_inputs.num_predicted_tokens
        current_num_predicted_tokens[idxs_to_merge] = num_predicted_tokens[:valid_batch_size]

        new_input = ContinuousBatchInput(
            input_ids=current_input_ids,
            position_ids=current_position_ids,
            num_valid_tokens=current_num_valid_tokens,
            num_predicted_tokens=current_num_predicted_tokens
        )

        return new_input, processed_outputs, idxs_to_merge

    def get_max_image_token_count(self, task_name: TaskNames) -> int:
        dummy_image = np.zeros(shape=(*self.tasks[task_name]["img_size"], 3))
        tiles, _ = self.processor._process_and_tile(dummy_image)
        num_image_tokens = tiles.shape[0] / self.processor.merge_size**2
        
        # Extra 1 to account for rotation token when present.
        return 1 + self.processor.num_register_tokens + int(num_image_tokens)

    def prediction_loop(
        self,
        images: List[np.ndarray],
        input_texts: List[str],
        task_names: List[TaskNames],
        batch_size: int | None = None,
        math_mode: bool = True,
        drop_repeated_tokens: bool = True,
        max_lookahead_tokens: Optional[int] = None,
        top_k: int = 5
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
        
        max_image_tokens = max(self.get_max_image_token_count(task) for task in set(task_names))
        self.setup_cache(batch_size, max_cache_len=max_image_tokens + self.model.config.sliding_window)

        batch_max_tokens = {}
        for idx, (img, txt, task) in enumerate(
            zip(images, input_texts, task_names)
        ):
            self.prompt_queue.append(
                FoundationPrompt(
                    id=idx, task_name=task, text=txt, image=img, math_mode=math_mode
                )
            )
            batch_max_tokens[idx] = (
                settings.FOUNDATION_MAX_TOKENS or self.tasks[task]["max_tokens"]
            )

        overall_max_tokens = max(batch_max_tokens.values())

        pbar = tqdm(
            total=len(self.prompt_queue),
            desc="Recognizing Text",
            disable=self.disable_tqdm,
        )

        batch_bboxes = torch.zeros(len(images), overall_max_tokens, 6)
        batch_pos = [0] * len(images)

        while self.prompt_queue or self.num_active_slots > 0:
            if (
                self.num_empty_slots / batch_size
            ) > self.min_prefill_ratio and self.prompt_queue:
                updated_inputs, outputs, merge_idxs = self.prefill(current_inputs, max_lookahead_tokens=max_lookahead_tokens)

                predicted_tokens_cpu = outputs.preds.cpu()
                scores_cpu = outputs.scores.cpu()
                token_probs_cpu = outputs.token_probs.cpu()
                for temp_idx, b_idx in enumerate(merge_idxs):
                    if self.batch_prompt_mapping[b_idx] is not None:
                        p_idx = self.batch_prompt_mapping[b_idx]
                        seq_len = predicted_tokens_cpu.shape[1]
                        for t_idx in range(seq_len):
                            token = predicted_tokens_cpu[temp_idx, t_idx].item()
                            predicted_tokens[p_idx].append(token)
                            batch_bboxes[p_idx, batch_pos[p_idx]] = outputs.bbox_preds[
                                temp_idx, t_idx
                            ]
                            batch_pos[p_idx] += 1
                            scores[p_idx].append(scores_cpu[temp_idx, t_idx].item())

                            top_k_probs, top_k_indices = torch.topk(
                                token_probs_cpu[temp_idx, t_idx], k=top_k, dim=-1
                            )
                            top_k_scores = {
                                top_k_indices[k].item(): top_k_probs[k].item()
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
                updated_inputs, outputs = self.decode(current_inputs, max_lookahead_tokens=max_lookahead_tokens)
                predicted_tokens_cpu = outputs.preds.cpu()
                scores_cpu = outputs.scores.cpu()
                token_probs_cpu = outputs.token_probs.cpu()

                for b_idx, p_idx in self.batch_prompt_mapping.items():
                    if p_idx is not None:
                        seq_len = predicted_tokens_cpu.shape[1]
                        should_stop = False

                        for t_idx in range(seq_len):
                            token = predicted_tokens_cpu[b_idx, t_idx].item()
                            predicted_tokens[p_idx].append(token)
                            batch_bboxes[p_idx, batch_pos[p_idx]] = outputs.bbox_preds[
                                b_idx, t_idx
                            ]
                            batch_pos[p_idx] += 1
                            scores[p_idx].append(scores_cpu[b_idx, t_idx].item())

                            top_k_probs, top_k_indices = torch.topk(
                                token_probs_cpu[temp_idx, t_idx], k=top_k, dim=-1
                            )
                            top_k_scores = {
                                top_k_indices[k].item(): top_k_probs[k].item()
                                for k in range(top_k)
                            }
                            topk_probs[p_idx].append(top_k_scores)

                            repeats = (
                                len(predicted_tokens[p_idx]) >= batch_max_tokens[p_idx]
                                or (drop_repeated_tokens and detect_repeat_token(predicted_tokens[p_idx]))
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
            mark_step()
        pbar.close()

        del self.kv_cache
        self.kv_cache = None
        torch.cuda.empty_cache()

        return predicted_tokens, batch_bboxes, scores, topk_probs