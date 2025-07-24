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


class ContinuousBatchingCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: int,
        dtype: int,
    ):
        # batch_size is deprecated in newer versions
        super().__init__(
            config,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=batch_size,
        )
        self.text_sliding_window = text_sliding_window
        self.num_layers = config.num_hidden_layers

        self.attention_mask = torch.zeros(
            (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
        )
        self.text_token_counts = [
            torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
            for _ in range(self.num_layers)
        ]

        self.dtype = dtype
        self.device = device

    def _shift_attention_mask_left(self, batch_idx: int, shift_amount: int):
        self.attention_mask[batch_idx, :-shift_amount] = self.attention_mask[
            batch_idx, shift_amount:
        ].clone()
        self.attention_mask[batch_idx, -shift_amount:] = 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = cache_kwargs.get("prefill", False)
        if prefill:
            return self._prefill_update(
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                key_states,
                value_states,
                cache_kwargs,
            )
        else:
            return self._decode_update(
                key_states, value_states, layer_idx, cache_kwargs
            )

    def update_text_counts(self, cache_idxs: List[int], new_text_lens: List[int]):
        assert len(cache_idxs) == len(new_text_lens)
        new_text_len_tensor = torch.tensor(
            new_text_lens, dtype=torch.long, device=self.device
        )
        for layer_idx in range(self.num_layers):
            self.text_token_counts[layer_idx][cache_idxs] = new_text_len_tensor

    """
    This matches the implemenation of the cache update, but needs to be called before the first
    cache update since the attention mask is used for other operations before the cache update
    """

    def maybe_shift_attention_mask(
        self, num_valid_tokens: List[int], cache_idxs: List[int]
    ):
        for batch_idx, cache_idx in enumerate(cache_idxs):
            new_text_len = num_valid_tokens[batch_idx]
            if new_text_len == 0:
                continue  # skip padded batch entry

            # Same token counts for all layers when we start, so we take 0th
            curr_text_cache_len = self.text_token_counts[0][cache_idx].item()

            if curr_text_cache_len + new_text_len <= self.text_sliding_window:
                # If we are under the sliding window length, shift the entire cache left
                # Since we setup the max cache length with enough buffer, this will ONLY drop
                # left padding tokens out
                shift = new_text_len
                self._shift_attention_mask_left(cache_idx, shift)
            else:
                # Shift entire cache left to make room for full text sliding window
                shift_amount = self.text_sliding_window - curr_text_cache_len
                # If this is <=0, we are already above the sliding window, so the attention mask stays the same
                if shift_amount > 0:
                    self._shift_attention_mask_left(cache_idx, shift_amount)

    # Mirrors the logic from _prefill_update
    def prefill_attention_mask_update(
        self,
        attention_mask: torch.Tensor,
        cache_idxs: List[int],
        text_lengths: List[int],
    ):
        seq_len = attention_mask.shape[1]

        for batch_idx, cache_idx in enumerate(cache_idxs):
            text_len = text_lengths[batch_idx]
            self.attention_mask[cache_idx] = 0  # Set default

            if text_len <= self.text_sliding_window:
                # This is safe since the cache length is larger than the max image tokens + sliding_window
                tokens_to_take = min(seq_len, self.max_cache_len)
                self.attention_mask[cache_idx, -tokens_to_take:] = attention_mask[
                    batch_idx, -tokens_to_take:
                ]
            else:
                # Place the last sliding_window text tokens at the end of cache
                cache_text_start = self.max_cache_len - self.text_sliding_window
                self.attention_mask[cache_idx, cache_text_start:] = 1

                # These include both image and padding tokens
                non_text_tokens = seq_len - text_len
                non_text_tokens_to_keep = min(
                    non_text_tokens, self.max_cache_len - self.text_sliding_window
                )

                # Take the last image_tokens_to_keep image tokens
                image_end_in_seq = seq_len - text_len

                self.attention_mask[
                    cache_idx,
                    cache_text_start - non_text_tokens_to_keep : cache_text_start,
                ] = attention_mask[
                    batch_idx,
                    image_end_in_seq - non_text_tokens_to_keep : image_end_in_seq,
                ]

    def _prefill_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_idxs: List[int] = cache_kwargs.get("cache_idxs", None)
        text_lengths: List[int] = cache_kwargs.get("text_lengths", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"
        assert text_lengths is not None, "text_lengths must be specified during prefill"

        # Get cache dimensions
        seq_len = key_states.shape[2]

        for batch_idx, cache_idx in enumerate(cache_idxs):
            text_len = text_lengths[batch_idx]

            if text_len <= self.text_sliding_window:
                # This is safe since the cache length is larger than the max image tokens + sliding_window
                tokens_to_take = min(seq_len, self.max_cache_len)
                start_idx = seq_len - tokens_to_take
                key_cache[cache_idx, :, -tokens_to_take:, :] = key_states[
                    batch_idx, :, start_idx:, :
                ]
                value_cache[cache_idx, :, -tokens_to_take:, :] = value_states[
                    batch_idx, :, start_idx:, :
                ]
            else:
                # Place the last sliding_window text tokens at the end of cache
                text_start_in_seq = seq_len - self.text_sliding_window
                cache_text_start = self.max_cache_len - self.text_sliding_window

                key_cache[cache_idx, :, cache_text_start:, :] = key_states[
                    batch_idx, :, text_start_in_seq:, :
                ]
                value_cache[cache_idx, :, cache_text_start:, :] = value_states[
                    batch_idx, :, text_start_in_seq:, :
                ]

                # These include both image and padding tokens
                non_text_tokens = seq_len - text_len
                non_text_tokens_to_keep = min(
                    non_text_tokens, self.max_cache_len - self.text_sliding_window
                )

                # Take the last image_tokens_to_keep image tokens
                image_end_in_seq = seq_len - text_len

                key_cache[
                    cache_idx,
                    :,
                    cache_text_start - non_text_tokens_to_keep : cache_text_start,
                    :,
                ] = key_states[
                    batch_idx,
                    :,
                    image_end_in_seq - non_text_tokens_to_keep : image_end_in_seq,
                    :,
                ]
                value_cache[
                    cache_idx,
                    :,
                    cache_text_start - non_text_tokens_to_keep : cache_text_start,
                    :,
                ] = value_states[
                    batch_idx,
                    :,
                    image_end_in_seq - non_text_tokens_to_keep : image_end_in_seq,
                    :,
                ]

        return key_states, value_states

    def _decode_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
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

        batch_size = key_states.shape[0]
        cache_idxs = list(range(batch_size))

        k_cache = self.key_cache[layer_idx]  # (B, H, L, D)
        v_cache = self.value_cache[layer_idx]  # (B, H, L, D)

        for batch_idx, cache_idx in enumerate(cache_idxs):
            new_text_len = num_valid_tokens[batch_idx].item()
            if new_text_len == 0:
                continue  # skip padded batch entry

            curr_text_cache_len = self.text_token_counts[layer_idx][cache_idx].item()

            # Decode is **left-padded** so we ignore these tokens
            k_new = key_states[batch_idx, :, -new_text_len:, :]
            v_new = value_states[batch_idx, :, -new_text_len:, :]

            if curr_text_cache_len + new_text_len <= self.text_sliding_window:
                # If we are under the sliding window length, shift the entire cache left
                # Since we setup the max cache length with enough buffer, this will ONLY drop
                # left padding tokens out
                shift = new_text_len
                k_cache[cache_idx, :, :-shift, :] = k_cache[
                    cache_idx, :, shift:, :
                ].clone()
                v_cache[cache_idx, :, :-shift, :] = v_cache[
                    cache_idx, :, shift:, :
                ].clone()
                k_cache[cache_idx, :, -shift:, :] = k_new
                v_cache[cache_idx, :, -shift:, :] = v_new

                self.text_token_counts[layer_idx][cache_idx] += new_text_len
            else:
                # Expand text region to exactly text_sliding_window tokens
                # Shift entire cache left to make room for the full sliding window

                # Calculate how much to shift left to accommodate full sliding window
                desired_text_start = self.max_cache_len - self.text_sliding_window

                # We need to figure out how many text tokens to keep and where to place them
                keep = self.text_sliding_window - new_text_len
                assert keep > 0, (
                    "Cannot add more new text tokens than the sliding window"
                )

                # Shift entire cache left to make room for full text sliding window
                shift_amount = self.text_sliding_window - curr_text_cache_len
                if shift_amount > 0:  # Cannot be negative, may be exactly 0
                    k_cache[cache_idx, :, :-shift_amount, :] = k_cache[
                        cache_idx, :, shift_amount:, :
                    ].clone()
                    v_cache[cache_idx, :, :-shift_amount, :] = v_cache[
                        cache_idx, :, shift_amount:, :
                    ].clone()

                # Now place the most recent 'keep' text tokens at the start of text region
                old_text_start = self.max_cache_len - curr_text_cache_len - shift_amount
                k_cache[
                    cache_idx, :, desired_text_start : desired_text_start + keep, :
                ] = k_cache[
                    cache_idx,
                    :,
                    old_text_start + (curr_text_cache_len - keep) : old_text_start
                    + curr_text_cache_len,
                    :,
                ].clone()
                v_cache[
                    cache_idx, :, desired_text_start : desired_text_start + keep, :
                ] = v_cache[
                    cache_idx,
                    :,
                    old_text_start + (curr_text_cache_len - keep) : old_text_start
                    + curr_text_cache_len,
                    :,
                ].clone()

                # Add new tokens at the end
                k_cache[
                    cache_idx, :, desired_text_start + keep : self.max_cache_len, :
                ] = k_new
                v_cache[
                    cache_idx, :, desired_text_start + keep : self.max_cache_len, :
                ] = v_new

                self.text_token_counts[layer_idx][cache_idx] = self.text_sliding_window

        self.key_cache[layer_idx] = k_cache
        self.value_cache[layer_idx] = v_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # This is used by HF models to determine the causal relationship between new tokens and cache
    # Our cache is left padded - So all tokens should always be visible to new tokens
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len
