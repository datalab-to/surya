"""llama-server launch flags (no llama.cpp: subprocess mocked)."""

from types import SimpleNamespace

from surya.inference.backends import llamacpp
from surya.settings import settings


def _cmd(monkeypatch, special):
    captured = {}
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "LLAMA_CPP_SPECIAL", special)
    monkeypatch.setattr(
        llamacpp, "_resolve_llama_server_binary", lambda: "llama-server"
    )
    monkeypatch.setattr(
        llamacpp, "_download_gguf_files", lambda: ("/m/model.gguf", "/m/mmproj.gguf")
    )

    class _Popen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = cmd
            self.pid = 1

    monkeypatch.setattr(llamacpp.subprocess, "Popen", _Popen)

    def fake_attach(
        backend,
        expected_model_name,
        spawn_fn,
        health_url_for,
        openai_url_for,
        startup_timeout,
        **kw,
    ):
        spawn_fn(8123)
        return SimpleNamespace(
            base_url="http://x/v1", model_name=expected_model_name, spawned_by_us=True
        )

    monkeypatch.setattr(llamacpp, "attach_or_spawn", fake_attach)
    llamacpp.LlamaCppBackend().start()
    return captured["cmd"]


def test_special_flag_default_on(monkeypatch):
    assert "--special" in _cmd(monkeypatch, True)


def test_special_flag_can_be_turned_off(monkeypatch):
    assert "--special" not in _cmd(monkeypatch, False)
