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


class StaticOpsContinuousBatchingLayerCache(StaticCache):
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
        cache_idxs: torch.tensor = cache_kwargs.get("cache_idxs", None)
        text_lengths: List[int] = cache_kwargs.get("text_lengths", None)
        cache_idx_length: int = cache_kwargs.get("cache_idxs_length", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"
        assert text_lengths is not None, "text_lengths must be specified during prefill"

        cache_idxs = cache_idxs[
            :cache_idx_length
        ]  # Ensure we only use the valid indices

        # Insert key and value states at the end of the cache
        new_tokens = key_states.shape[2]

        # Direct right-aligned assignment
        key_cache[cache_idxs, :, -new_tokens:] = key_states[:cache_idx_length]
        value_cache[cache_idxs, :, -new_tokens:] = value_states[:cache_idx_length]

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
        # (B, H, L, D)
        max_valid_tokens = num_valid_tokens.max().item()

        self.key_cache = torch.roll(self.key_cache, -max_valid_tokens, dims=2)
        self.value_cache = torch.roll(self.value_cache, -max_valid_tokens, dims=2)

        new_k = key_states[:, :, -max_valid_tokens:, :]
        new_v = value_states[:, :, -max_valid_tokens:, :]

        self.key_cache[:, :, -max_valid_tokens:, :] = new_k
        self.value_cache[:, :, -max_valid_tokens:, :] = new_v
        return self.key_cache, self.value_cache


class StaticOpsContinuousBatchingCache:
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
            StaticOpsContinuousBatchingLayerCache(
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
        max_valid_tokens = num_valid_tokens.max().item()
        if max_valid_tokens == 0:
            # If no valid tokens, we don't need to update the attention mask
            return

        # Shift the attention mask to the left by max_valid_tokens
        self.attention_mask = self.attention_mask.roll(-1 * max_valid_tokens, dims=1)
        self.attention_mask[:, -max_valid_tokens:] = 0  # Mask out all new tokens

        seq_len = self.attention_mask.shape[1]
        positions = torch.arange(seq_len, device=self.attention_mask.device).unsqueeze(
            0
        )

        # Since cache_idxs is padded, num_valid_tokens should also be padded with zeros
        # for inactive positions, so we can process the full batch uniformly
        valid_mask = (positions >= (seq_len - num_valid_tokens.unsqueeze(1))).to(
            dtype=self.attention_mask.dtype
        )

        # Update the attention mask for the current batch elements
        self.attention_mask = self.attention_mask | valid_mask

    # Mirrors the logic from _prefill_update
    def prefill_attention_mask_update(
        self,
        attention_mask: torch.Tensor,
        cache_idxs: torch.Tensor,
        text_lengths: List[int],
        image_lengths: List[int],
        cache_idxs_length: Optional[int] = None,
    ):
        # Set from -(image_length + text_length) to end to 1 for each batch element
        seq_len = attention_mask.shape[1]
        self.attention_mask[cache_idxs] = (
            0  # Reset the attention mask for the current batch elements
        )
        self.attention_mask[cache_idxs, -seq_len:] = attention_mask[
            : cache_idxs.size(0)
        ]

    # This is used by HF models to determine the causal relationship between new tokens and cache
    # Our cache is left padded - So all tokens should always be visible to new tokens
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len
