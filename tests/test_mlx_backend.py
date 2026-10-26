"""MLX backend launch and attach logic (no mlx, no model: subprocess and probes mocked)."""

from types import SimpleNamespace

import pytest

from surya.inference import _build_backend
from surya.inference.backends import mlx as mlx_backend
from surya.inference.backends.spawn import SpawnError
from surya.settings import settings


def test_build_backend_mlx():
    assert isinstance(_build_backend("mlx"), mlx_backend.MlxBackend)


def test_start_requires_model(monkeypatch):
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "SURYA_MLX_MODEL", None)
    with pytest.raises(SpawnError, match="SURYA_MLX_MODEL"):
        mlx_backend.MlxBackend().start()


def test_spawn_command(monkeypatch):
    captured = {}
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "SURYA_MLX_MODEL", "/m/surya-mlx-int8")
    monkeypatch.setattr(settings, "MLX_VLM_EXTRA_ARGS", "--max-num-seqs 4 --foo 'a b'")
    monkeypatch.setattr(mlx_backend, "_require_mlx_vlm", lambda: None)

    class _Popen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = cmd
            self.pid = 1

    monkeypatch.setattr(mlx_backend.subprocess, "Popen", _Popen)

    def fake_attach(
        backend,
        expected_model_name,
        spawn_fn,
        health_url_for,
        openai_url_for,
        startup_timeout,
        **kw,
    ):
        captured["expected"] = expected_model_name
        spawn_fn(8123)
        return SimpleNamespace(
            base_url="http://x/v1", model_name=expected_model_name, spawned_by_us=True
        )

    monkeypatch.setattr(mlx_backend, "attach_or_spawn", fake_attach)
    handle = mlx_backend.MlxBackend().start()
    cmd = captured["cmd"]
    assert (
        cmd[1:3] == ["-m", "mlx_vlm.server"]
        and cmd[cmd.index("--model") + 1] == "/m/surya-mlx-int8"
    )
    assert cmd[cmd.index("--port") + 1] == "8123" and "--max-num-seqs" in cmd
    assert cmd[cmd.index("--foo") + 1] == "a b"
    assert (
        captured["expected"] == "/m/surya-mlx-int8"
        and handle.model_name == "/m/surya-mlx-int8"
    )


def test_attach_external_uses_server_model_id(monkeypatch):
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setattr(
        mlx_backend, "probe_health", lambda url: url == "http://127.0.0.1:9000"
    )
    monkeypatch.setattr(mlx_backend, "probe_model_id", lambda base: "/some/mlx/dir")
    h = mlx_backend.MlxBackend().start()
    assert (
        h.base_url == "http://127.0.0.1:9000/v1"
        and h.model_name == "/some/mlx/dir"
        and not h.spawned_by_us
    )


def test_attach_external_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setattr(mlx_backend, "probe_health", lambda url: False)
    with pytest.raises(SpawnError, match="not reachable"):
        mlx_backend.MlxBackend().start()


def test_attach_external_rejects_model_mismatch_and_missing_id(monkeypatch):
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setattr(mlx_backend, "probe_health", lambda url: True)
    monkeypatch.setattr(settings, "SURYA_MLX_MODEL", "/m/surya-mlx-int8")
    monkeypatch.setattr(mlx_backend, "probe_model_id", lambda base: "/other/model")
    with pytest.raises(SpawnError, match="serves"):
        mlx_backend.MlxBackend().start()
    monkeypatch.setattr(mlx_backend, "probe_model_id", lambda base: None)
    with pytest.raises(SpawnError, match="did not report"):
        mlx_backend.MlxBackend().start()
