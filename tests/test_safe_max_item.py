import torch
import pytest

from surya.common.util import safe_max_item


def _get_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return [torch.device(d) for d in devices]


@pytest.fixture(params=_get_devices(), ids=lambda d: d.type)
def device(request):
    return request.param


class TestSafeMaxItem:
    """Tests for safe_max_item — MPS-safe replacement for tensor.max().item()."""

    def test_single_element(self, device):
        t = torch.tensor([42], device=device)
        assert safe_max_item(t) == 42

    def test_multiple_elements(self, device):
        t = torch.tensor([1, 5, 3, 2, 4], device=device)
        assert safe_max_item(t) == 5

    def test_negative_values(self, device):
        t = torch.tensor([-10, -3, -7], device=device)
        assert safe_max_item(t) == -3

    def test_large_tensor(self, device):
        """Larger tensors are more likely to trigger the MPS kernel bug."""
        t = torch.arange(8192, device=device)
        assert safe_max_item(t) == 8191

    def test_cumulative_seqlen_pattern(self, device):
        """Mimics the exact pattern from unpack_qkv_with_mask."""
        # Simulate cu_seqlens for variable-length sequences
        seq_lens = [128, 256, 64, 512, 196]
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        result = safe_max_item(seq_lengths)
        assert result == 512

    def test_returns_python_int(self, device):
        t = torch.tensor([1, 2, 3], device=device)
        result = safe_max_item(t)
        assert isinstance(result, int)

    def test_float_tensor(self, device):
        t = torch.tensor([1.5, 3.7, 2.1], device=device)
        result = safe_max_item(t)
        assert result == pytest.approx(3.7)
