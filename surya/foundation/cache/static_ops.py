from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import PretrainedConfig

from surya.foundation.cache.dynamic_ops import DynamicOpsCache

"""
Special cache class for the surya foundation model that supports - 
1) Static shape
2) A custom sliding window, where image tokens stay in cache, and text tokens are popped
3) Continuous batching - merging etc
4) Attention mask management - To match with what's currently in the cache

Heavily inspired from https://github.com/huggingface/transformers/blob/0725cd6953803b8aacfc85288cbfb83dea30c469/src/transformers/cache_utils.py#L1079
"""


class StaticOpsCache(DynamicOpsCache):
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: int,
        dtype: int,
    ):
        self.text_sliding_window = text_sliding_window
        self.num_layers = config.num_hidden_layers
        self.max_batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        # Cache init is taken from huggingface StaticCache - https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/cache_utils.py#L1125
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        device = torch.device(device) if device is not None else None
        for _ in range(config.num_hidden_layers):
            new_layer_key_cache = torch.zeros(
                cache_shape, dtype=self._dtype, device=device
            )
            new_layer_value_cache = torch.zeros(
                cache_shape, dtype=self._dtype, device=device
            )
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

        self.attention_mask = torch.zeros(
            (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
        )
        self.text_token_counts = [
            torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
            for _ in range(self.num_layers)
        ]

        self.dtype = dtype
        self.device = device

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = cache_kwargs.get("prefill", False)
        update_fn = self._prefill_update if prefill else self._decode_update
        return update_fn(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            self.text_token_counts[layer_idx],
            cache_kwargs,
        )

    def _prefill_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        text_token_counts: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_idxs: torch.tensor = cache_kwargs.get("cache_idxs", None)
        text_lengths: List[int] = cache_kwargs.get("text_lengths", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"
        assert text_lengths is not None, "text_lengths must be specified during prefill"

        cache_idx_length = len(cache_idxs)
        full_batch = len(cache_idxs) == self.max_batch_size

        # Insert key and value states at the end of the cache
        new_tokens = key_states.shape[2]

        # Direct right-aligned assignment
        if full_batch:
            key_cache[:, :, -new_tokens:] = key_states
            value_cache[:, :, -new_tokens:] = value_states
        else:
            key_cache[cache_idxs, :, -new_tokens:] = key_states[:cache_idx_length]
            value_cache[cache_idxs, :, -new_tokens:] = value_states[:cache_idx_length]

        return key_states, value_states

    # """
    # Matches the logic of the decode update, but needs to be called before the updates
    # since some parts of the model depend on the attention mask
    # """
    def decode_attention_mask_update(
        self, num_valid_tokens: torch.Tensor, cache_idxs: List[int]
    ):
        max_valid_tokens = num_valid_tokens.max().item()
        if max_valid_tokens == 0:
            # If no valid tokens, we don't need to update the attention mask
            return

        # Shift the attention mask to the left by max_valid_tokens
        self.attention_mask = self.attention_mask.roll(-1 * max_valid_tokens, dims=1)
        self.attention_mask[:, -max_valid_tokens:] = (
            1  # Full attention to all new tokens
        )

    # Mirrors the logic from _prefill_update
    def prefill_attention_mask_update(
        self,
        attention_mask: torch.Tensor,
        merge_idxs: torch.Tensor,
        valid_batch_size: torch.Tensor,
        text_lengths: List[int],
    ):
        # Set from -(image_length + text_length) to end to 1 for each batch element
        seq_len = attention_mask.shape[1]
        self.attention_mask[merge_idxs] = (
            0  # Reset the attention mask for the current batch elements
        )
        self.attention_mask[merge_idxs, -seq_len:] = attention_mask[:valid_batch_size]

    def _decode_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        text_token_counts: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Naive, always assumes we'll roll by a fixed amount
        # Needs left padding with beacons to work properly

        num_valid_tokens: torch.Tensor = cache_kwargs.get(
            "num_valid_tokens"
        )  # shape: (B,)
        assert num_valid_tokens is not None, (
            "`num_valid_tokens` must be provided in `cache_kwargs`"
        )
        # (B, H, L, D)

        valid_tokens = key_states.shape[2]

        key_cache.copy_(torch.roll(key_cache, -valid_tokens, dims=2))
        value_cache.copy_(torch.roll(value_cache, -valid_tokens, dims=2))

        key_cache[:, :, -valid_tokens:, :] = key_states
        value_cache[:, :, -valid_tokens:, :] = value_states

        # In-place edit - Mutates
        text_token_counts += num_valid_tokens
        text_token_counts.clamp_(max=self.text_sliding_window)
        return key_cache, value_cache

    # The attention mask managed by our kv cache automatically masks the tokens
    # in the cache, so we can return full length for HF to use in other places
    # This is mainly utilized in the cache_positions creation
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len
