import pytest
import torch

from surya.common.util import device_matches


@pytest.mark.parametrize(
    "device,name,expected",
    [
        # String forms without index
        ("cuda", "cuda", True),
        ("cpu", "cpu", True),
        ("mps", "mps", True),
        ("xla", "xla", True),
        # Indexed string forms — the #500 bug
        ("cuda:0", "cuda", True),
        ("cuda:1", "cuda", True),
        ("cuda:7", "cuda", True),
        ("mps:0", "mps", True),
        # torch.device without index
        (torch.device("cuda"), "cuda", True),
        (torch.device("cpu"), "cpu", True),
        (torch.device("mps"), "mps", True),
        # torch.device with index
        (torch.device("cuda", 0), "cuda", True),
        (torch.device("cuda", 1), "cuda", True),
        (torch.device("cuda:2"), "cuda", True),
        # Mismatches
        ("cpu", "cuda", False),
        ("cuda", "cpu", False),
        ("cuda:0", "cpu", False),
        (torch.device("cuda"), "cpu", False),
        (torch.device("cuda", 0), "mps", False),
        # None
        (None, "cuda", False),
        (None, "cpu", False),
    ],
)
def test_device_matches(device, name, expected):
    assert device_matches(device, name) is expected


def test_device_matches_distinguishes_prefix_collision():
    # "cuda0" should not match name "cuda" — only "cuda" or "cuda:*" counts.
    assert device_matches("cuda0", "cuda") is False
    assert device_matches("cudaxx", "cuda") is False
