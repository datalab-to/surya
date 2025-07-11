from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import StaticCache
from transformers import PretrainedConfig

"""
Special cache class for the surya foundation model that supports - 
1) Static shape
2) A custom sliding window, where image tokens stay in cache, and text tokens are popped
3) Continuous batching - merging etc
4) Attention mask management - To match with what's currently in the cache

Heavily inspired from https://github.com/huggingface/transformers/blob/0725cd6953803b8aacfc85288cbfb83dea30c469/src/transformers/cache_utils.py#L1079
"""


class ContinuousBatchingLayerCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # No need for the super class call, it just overwrites the caches
        # At some point, we should consider not inheriting from StaticCache
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size

        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        self.text_sliding_window = text_sliding_window
        self.text_token_counts = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=device
        )
        self.cache_image_end = self.max_cache_len - self.text_sliding_window

        self.dtype = dtype
        self.device = device

    # This is used by HF models to determine the causal relationship between new tokens and cache
    # Our cache is left padded - So all tokens should always be visible to new tokens
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = cache_kwargs.get("prefill", False)
        if prefill:
            return self._prefill_update(
                self.key_cache,
                self.value_cache,
                key_states,
                value_states,
                cache_kwargs,
            )
        else:
            return self._decode_update(key_states, value_states, cache_kwargs)

    def _prefill_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_idxs: torch.Tensor = cache_kwargs.get("cache_idxs", None)
        cache_idx_length: int = cache_kwargs.get("cache_idx_length", None)
        text_lengths: torch.Tensor = cache_kwargs.get("text_lengths", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"
        assert text_lengths is not None, "text_lengths must be specified during prefill"

        # Get cache dimensions
        seq_len = key_states.shape[2]
        text_lengths = text_lengths[:cache_idx_length]

        # Calculate dimensions for each batch item
        state_image_ends = seq_len - text_lengths
        cache_image_starts = self.cache_image_end - state_image_ends

        # Process image tokens
        for i in range(cache_idx_length):
            cache_idx = cache_idxs[i]
            state_image_end = state_image_ends[i]
            cache_image_start = cache_image_starts[i]

            # Create index tensors for the cache positions
            cache_seq_indices = torch.arange(
                cache_image_start, self.cache_image_end, device=key_states.device
            )
            state_seq_indices = torch.arange(
                0, state_image_end, device=key_states.device
            )

            # Use index_select to get the source data and index_copy_ to place it
            selected_keys = torch.index_select(
                key_states[i], dim=1, index=state_seq_indices
            )
            selected_values = torch.index_select(
                value_states[i], dim=1, index=state_seq_indices
            )

            key_cache[cache_idx].index_copy_(
                dim=1, index=cache_seq_indices, source=selected_keys
            )
            value_cache[cache_idx].index_copy_(
                dim=1, index=cache_seq_indices, source=selected_values
            )

        # Process text tokens
        for i in range(cache_idx_length):
            cache_idx = cache_idxs[i]
            text_len = text_lengths[i]
            cache_text_len = min(text_len, self.text_sliding_window)
            cache_text_end = self.cache_image_end + cache_text_len

            # Create index tensors for text tokens
            cache_text_indices = torch.arange(
                self.cache_image_end, cache_text_end, device=key_states.device
            )
            state_text_indices = torch.arange(
                seq_len - cache_text_len, seq_len, device=key_states.device
            )

            # Use index_select to get the source data and index_copy_ to place it
            selected_keys = torch.index_select(
                key_states[i], dim=1, index=state_text_indices
            )
            selected_values = torch.index_select(
                value_states[i], dim=1, index=state_text_indices
            )

            key_cache[cache_idx].index_copy_(
                dim=1, index=cache_text_indices, source=selected_keys
            )
            value_cache[cache_idx].index_copy_(
                dim=1, index=cache_text_indices, source=selected_values
            )

            self.text_token_counts[cache_idx] = cache_text_len

        return key_states, value_states

    def _decode_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Static cache update
        - respects per-batch text token limits
        - per-batch valid token lengths (right-padded inputs)

        kv states are expected to have shape [batch_size, kv_heads, T_pad, head_dim]
        They may have different `true` lengths, to account for multi token preds, or beacon tokens
        Expects `num_valid_tokens` in cache_kwargs: a tensor of shape (B,) indicating the number
        of actual (non-padded) tokens to add per batch element.
        """

        num_valid_tokens: torch.Tensor = cache_kwargs.get(
            "num_valid_tokens"
        )  # shape: (B,)
        assert num_valid_tokens is not None, (
            "`num_valid_tokens` must be provided in `cache_kwargs`"
        )

        k_cache = self.key_cache  # (B, H, L, D)
        v_cache = self.value_cache  # (B, H, L, D)

        existing_space = (
            self.text_sliding_window - self.text_token_counts
        )  # free room per batch
        end_insert_pos = (
            self.text_sliding_window + self.cache_image_end - num_valid_tokens
        )

        shifts = (num_valid_tokens - existing_space).clamp(min=0)
        # ─► 0  when enough space
        # ─► new_token_count-existing_space when partial space
        # ─► new_token_count when no space

        insert_positions = torch.where(
            existing_space >= num_valid_tokens,  # “no-shift” case
            self.text_token_counts + self.cache_image_end,  # append after current text
            end_insert_pos,
        )

        shifts = shifts.view(-1, 1, 1).expand(
            -1, k_cache.shape[1], self.text_sliding_window
        )

        # Shift the text tokens if needed
        # Always shifting slows things down
        if shifts.sum().item() > 0:
            indices = (
                torch.arange(self.text_sliding_window, device=k_cache.device)
                .view(1, 1, -1)
                .expand(k_cache.shape[0], k_cache.shape[1], -1)
            )
            shifted_indices = (
                (indices + shifts).clamp(0, self.text_sliding_window - 1).unsqueeze(-1)
            )
            shifted_indices = shifted_indices.expand(-1, -1, -1, k_cache.shape[-1])
            v_cache[:, :, self.cache_image_end :, :] = torch.gather(
                v_cache[:, :, self.cache_image_end :, :], dim=2, index=shifted_indices
            )
            k_cache[:, :, self.cache_image_end :, :] = torch.gather(
                k_cache[:, :, self.cache_image_end :, :], dim=2, index=shifted_indices
            )

        for batch_idx, new_text_len in enumerate(num_valid_tokens):
            cache_idx = batch_idx
            new_text_len = num_valid_tokens[batch_idx]

            insert_position = insert_positions[batch_idx]
            end_position = insert_position + new_text_len

            k_new = key_states[batch_idx, :, -new_text_len:, :]
            v_new = value_states[batch_idx, :, -new_text_len:, :]

            if new_text_len > 0:
                k_cache[cache_idx, :, insert_position:end_position] = k_new
                v_cache[cache_idx, :, insert_position:end_position] = v_new
                self.text_token_counts[cache_idx] += new_text_len

        self.text_token_counts.clamp_(max=self.text_sliding_window)

        # These were passed in as self.key_cache and self.value_cache, so we return them to match the interface
        return k_cache, v_cache


class ContinuousBatchingCache:
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
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size
        self.cache_image_end = self.max_cache_len - self.text_sliding_window

        self.attention_mask = torch.zeros(
            (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
        )
        self.layer_caches = [
            ContinuousBatchingLayerCache(
                config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                text_sliding_window=text_sliding_window,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.num_layers)
        ]

        self.dtype = dtype
        self.device = device

    def decode_attention_mask_update(
        self, num_valid_tokens: torch.Tensor, cache_idxs: List[int]
    ):
        num_valid_tokens_list = num_valid_tokens.tolist()
        for batch_idx, cache_idx in enumerate(cache_idxs):
            existing_token_count = self.layer_caches[0].text_token_counts[cache_idx]
            new_text_len = num_valid_tokens_list[batch_idx]

            cache_text_len = min(
                self.text_sliding_window, new_text_len + existing_token_count
            )
            cache_text_end = self.cache_image_end + cache_text_len

            # Use index_copy_ instead of slicing
            indices = torch.arange(
                self.cache_image_end, cache_text_end, device=self.attention_mask.device
            )
            ones = torch.ones(
                indices.size(0),
                dtype=self.attention_mask.dtype,
                device=self.attention_mask.device,
            )
            self.attention_mask[cache_idx].index_copy_(
                dim=0, index=indices, source=ones
            )

    # Mirrors the logic from _prefill_update
    def prefill_attention_mask_update(
        self,
        attention_mask: torch.Tensor,
        cache_idxs: torch.Tensor,
        text_lengths: torch.Tensor,
        image_lengths: torch.Tensor,
        cache_idxs_length: torch.Tensor,
    ):
        for batch_idx, cache_idx in enumerate(cache_idxs[:cache_idxs_length]):
            text_len = text_lengths[batch_idx]
            image_len = image_lengths[batch_idx]
            valid_mask_length = text_len + image_len
            self.attention_mask[cache_idx] = 0  # Set default

            # Assign image mask using index_copy_
            cache_image_start = self.cache_image_end - image_len
            image_indices = torch.arange(
                cache_image_start,
                self.cache_image_end,
                device=self.attention_mask.device,
            )
            image_mask_values = attention_mask[batch_idx, -valid_mask_length:-text_len]
            self.attention_mask[cache_idx].index_copy_(
                dim=0, index=image_indices, source=image_mask_values
            )

            # Assign text mask using index_copy_
            cache_text_length = min(text_len, self.text_sliding_window)
            cache_text_end = self.cache_image_end + cache_text_length
            text_indices = torch.arange(
                self.cache_image_end, cache_text_end, device=self.attention_mask.device
            )
            text_mask_values = attention_mask[batch_idx, -cache_text_length:]
            self.attention_mask[cache_idx].index_copy_(
                dim=0, index=text_indices, source=text_mask_values
            )

    # This is used by HF models to determine the causal relationship between new tokens and cache
    # Our cache is left padded - So all tokens should always be visible to new tokens
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len
