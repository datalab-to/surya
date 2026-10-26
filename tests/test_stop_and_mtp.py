"""Unit tests (no GPU, no VLM backend) for stop forwarding and MTP auto-detect."""

import json
from types import SimpleNamespace

from PIL import Image

from surya.inference.backends import openai_client, vllm
from surya.inference.schema import BatchInputItem
from surya.settings import settings


class _Capture:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content="<div>x</div>")
        ch = SimpleNamespace(message=msg, logprobs=None)
        return SimpleNamespace(choices=[ch], usage=SimpleNamespace(completion_tokens=3))


def _run(monkeypatch, **overrides):
    for k, v in overrides.items():
        monkeypatch.setattr(settings, k, v)
    cap = _Capture()
    item = BatchInputItem(
        image=Image.new("RGB", (64, 64)), prompt="p", prompt_type="block"
    )
    openai_client._generate_one(item, cap, "m", 16, 0.0, 0.1, 5.0, False)
    return cap.kwargs


def test_no_stop_by_default(monkeypatch):
    kw = _run(monkeypatch, SURYA_STOP_STRINGS=None, SURYA_STOP_TOKEN_IDS=None)
    assert "stop" not in kw and "stop_token_ids" not in kw.get("extra_body", {})


def test_stop_forwarded(monkeypatch):
    kw = _run(monkeypatch, SURYA_STOP_STRINGS="<|im_end|>", SURYA_STOP_TOKEN_IDS="2, 3")
    assert kw["stop"] == ["<|im_end|>"]
    assert kw["extra_body"]["stop_token_ids"] == [2, 3]


def _fake_hf(tmp_path, cfg):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return lambda *a, **k: str(p)


def test_mtp_detect(monkeypatch, tmp_path):
    import huggingface_hub

    surya2 = {
        "num_nextn_predict_layers": 1,
        "text_config": {"mtp_num_hidden_layers": 1},
    }
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf(tmp_path, surya2))
    assert vllm._checkpoint_has_mtp_head("x/y") is True
    none = {"text_config": {"mtp_num_hidden_layers": 0}}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf(tmp_path, none))
    assert vllm._checkpoint_has_mtp_head("x/y") is False


def test_mtp_explicit_overrides_detect(monkeypatch):
    monkeypatch.setattr(settings, "VLLM_ENABLE_MTP", False)
    assert vllm._mtp_enabled() is False
    monkeypatch.setattr(settings, "VLLM_ENABLE_MTP", True)
    assert vllm._mtp_enabled() is True


def test_mtp_detect_failure_is_off(monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(settings, "VLLM_ENABLE_MTP", None)
    monkeypatch.setattr(settings, "SURYA_MODEL_CHECKPOINT", "no/such-local-path")
    assert vllm._mtp_enabled() is False


def test_stop_ids_not_sent_when_backend_disallows(monkeypatch):
    for k, v in dict(SURYA_STOP_STRINGS=None, SURYA_STOP_TOKEN_IDS="2").items():
        monkeypatch.setattr(settings, k, v)
    cap = _Capture()
    item = BatchInputItem(
        image=Image.new("RGB", (64, 64)), prompt="p", prompt_type="block"
    )
    openai_client._generate_one(
        item, cap, "m", 16, 0.0, 0.1, 5.0, False, allow_stop_token_ids=False
    )
    assert "stop_token_ids" not in cap.kwargs.get("extra_body", {})


def test_mtp_detect_malformed_config_is_off(monkeypatch, tmp_path):
    import huggingface_hub

    p = tmp_path / "config.json"
    p.write_text("{not json")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(p))
    assert vllm._checkpoint_has_mtp_head("x/y") is False
    p.write_text(json.dumps({"num_nextn_predict_layers": "1", "text_config": []}))
    assert vllm._checkpoint_has_mtp_head("x/y") is False


def test_llama_gate_applies_to_retries(monkeypatch):
    """allow_stop_token_ids=False must hold for the first request and for the repeat-triggered retries."""
    from surya.inference.backends.openai_client import chat_completions_batch

    monkeypatch.setattr(settings, "SURYA_STOP_TOKEN_IDS", "2")
    monkeypatch.setattr(settings, "SURYA_STOP_STRINGS", "<|im_end|>")
    seen = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kw):
            seen.append(kw)
            ch = SimpleNamespace(
                message=SimpleNamespace(
                    content="a b a b a b a b a b a b a b a b a b a b a b a b"
                ),
                logprobs=None,
                finish_reason="stop",
            )
            return SimpleNamespace(
                choices=[ch], usage=SimpleNamespace(completion_tokens=3)
            )

    item = BatchInputItem(
        image=Image.new("RGB", (64, 64)), prompt="p", prompt_type="block"
    )
    monkeypatch.setattr(openai_client.time, "sleep", lambda s: None)
    chat_completions_batch(
        [item], Client(), "m", allow_stop_token_ids=False, max_retries=2
    )
    assert seen and all("stop_token_ids" not in kw.get("extra_body", {}) for kw in seen)
    assert all(
        kw.get("stop") == ["<|im_end|>"] for kw in seen
    )  # string stops stay supported


def test_guided_json_gate(monkeypatch):
    """allow_guided_json=False keeps response_format off the request; the default still sends it."""
    from surya.inference.backends.openai_client import chat_completions_batch

    seen = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kw):
            seen.append(kw)
            ch = SimpleNamespace(
                message=SimpleNamespace(content="[]"),
                logprobs=None,
                finish_reason="stop",
            )
            return SimpleNamespace(
                choices=[ch], usage=SimpleNamespace(completion_tokens=1)
            )

    item = BatchInputItem(
        image=Image.new("RGB", (64, 64)),
        prompt="p",
        prompt_type="layout",
        guided_json={"type": "array"},
    )
    chat_completions_batch([item], Client(), "m", allow_guided_json=False)
    chat_completions_batch([item], Client(), "m")
    assert "response_format" not in seen[0]
    assert seen[1]["response_format"]["type"] == "json_schema"


def test_mtp_local_malformed_config_is_off(monkeypatch, tmp_path):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    (tmp_path / "config.json").write_text("{not json")
    assert vllm._checkpoint_has_mtp_head(str(tmp_path)) is False
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 1}})
    )
    assert vllm._checkpoint_has_mtp_head(str(tmp_path)) is True


def test_trailing_end_marker_is_stripped():
    from surya.inference.backends.openai_client import _strip_end_marker

    assert _strip_end_marker("<p>a</p><|im_end!>") == "<p>a</p>"
    assert _strip_end_marker("<p>a</p>\n<|im_end|>\n") == "<p>a</p>"
    assert _strip_end_marker("<p>a</p>") == "<p>a</p>"
    assert _strip_end_marker("a <|im_end|> b") == "a <|im_end|> b"
