"""The vllm launcher builds the docker command the package users get: image tag and MTP flag (no GPU, docker mocked)."""

import json
from types import SimpleNamespace

from surya.inference.backends import vllm
from surya.settings import settings


def _argv(monkeypatch, has_mtp):
    captured = {}
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "VLLM_ENABLE_MTP", None)
    monkeypatch.setattr(vllm, "_checkpoint_has_mtp_head", lambda ckpt: has_mtp)
    monkeypatch.setattr(vllm.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        vllm.subprocess,
        "run",
        lambda cmd, **kw: captured.setdefault("cmd", cmd)
        and SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

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

    monkeypatch.setattr(vllm, "attach_or_spawn", fake_attach)
    vllm.VllmBackend().start()
    return captured["cmd"]


def test_launcher_image_and_no_mtp(monkeypatch):
    cmd = _argv(monkeypatch, has_mtp=False)
    assert "vllm/vllm-openai:v0.25.1" in cmd
    assert "--speculative-config" not in cmd


def test_launcher_mtp_when_head_present(monkeypatch):
    cmd = _argv(monkeypatch, has_mtp=True)
    i = cmd.index("--speculative-config")
    assert json.loads(cmd[i + 1]) == {
        "method": "mtp",
        "num_speculative_tokens": settings.VLLM_MTP_TOKENS,
    }
