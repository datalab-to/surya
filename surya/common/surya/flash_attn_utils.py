from typing import Optional
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
from flash_attn.bert_padding import index_first_axis as _index_first_axis
from flash_attn.bert_padding import pad_input

def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    from surya.common.util import safe_max_item
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = safe_max_item(seqlens_in_batch)
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    query_length: int,
    indices_k,
    cu_seqlens_k,
    max_seqlen_in_batch_k
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = _index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = _index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        raise NotImplementedError()

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

def flash_attn_prefill(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float,
    scaling: float,
    query_length: int,
    batch_size: int,
    indices_k: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_in_batch_k: int,
    **kwargs
):
    """
    Wrapper for flash attention during the prefill stage
    query_states must have shape (batch_size, num_heads, seq_len, head_dim)
    key_states and value_states must have shape (batch_size, num_kv_heads, kv_len, head_dim)

    This is the opposite of what is required by flash attention, but keeps parity with the HF convention

    query_length, batch_size, indices_k, cu_seqlens_k, and max_seqlen_in_batch_k should come from the flash attention kwargs
    """
    query_states, key_states, value_states = query_states.transpose(1,2), key_states.transpose(1,2), value_states.transpose(1,2)
    q_flash, k_flash, v_flash, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
        query_states, key_states, value_states, query_length, indices_k, cu_seqlens_k, max_seqlen_in_batch_k
    )
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    # Returning None for attn_weights to match other attention interfaces
    flash_attn_out = _flash_attn_varlen_func(
        q_flash,
        k_flash,
        v_flash,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=module.is_causal,
    )
    return pad_input(flash_attn_out, indices_q, batch_size, query_length), None

# NOTE: Does not support dropout, accepts argument as kwargs to maintain compatibility
# This function is an order of magnitude faster than the prefill variant, or using the HF interface
def flash_attn_decode(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    **kwargs,
):
    """
    Wrapper for flash attention during the decode stage
    
    query_states must have shape (batch_size, num_heads, seq_len, head_dim), 1 is the seq length in the decoding stage
    key_states and value_states must have shape (batch_size, num_kv_heads, kv_len, head_dim)

    This is the opposite of what is required by flash attention, but keeps parity with the HF convention

    This function computes the left pad and cache seqlens to pass into FA2. For example - 
    Given an attention_mask shaped (batch_size=2, seq_len=8), where 0 = padding, 1 = real token
    attention_mask =
    tensor([
        [0, 0, 1, 1, 1, 0, 0, 0],  # ← batch 0
        [0, 1, 1, 1, 1, 1, 1, 0],  # ← batch 1
    ])
    cache_leftpad = tensor([2, 1], dtype=torch.int32)
    cache_seqlens = tensor([5, 7], dtype=torch.int32)
    These values allow FlashAttention to use a static cache layout with efficient slicing during decoding.
    """
    query_states, key_states, value_states = query_states.transpose(1,2), key_states.transpose(1,2), value_states.transpose(1,2)

    cache_leftpad = (attention_mask == 0).cumprod(dim=1).sum(dim=1).to(torch.int32)
    cache_seqlens = (attention_mask * torch.arange(attention_mask.size(1), device=attention_mask.device)).argmax(dim=1).to(torch.int32) + 1

    # Returning None for attn_weights to match other attention interfaces
    return _flash_attn_with_kvcache(
        q=query_states,
        k_cache=key_states,
        v_cache=value_states,
        cache_leftpad=cache_leftpad,
        cache_seqlens=cache_seqlens,
        causal=module.is_causal,
        softmax_scale=scaling,
    ), None