"""The vllm launcher builds the container command the package users get: image tag,
MTP flag and the docker/podman GPU flags (no GPU, container runtime mocked)."""

import json
from types import SimpleNamespace

import pytest

from surya.inference.backends import spawn, vllm
from surya.settings import settings


def _argv(
    monkeypatch,
    has_mtp=False,
    binaries=("docker",),
    version_banner=None,
    runtime_setting=None,
    gpus="0",
):
    """Run VllmBackend.start() with the runtime mocked out; return the spawn argv.

    `binaries` are the names shutil.which finds (resolved to /usr/bin/<name>);
    `version_banner` is what `<binary> --version` prints (defaults to the name).
    """
    captured = {}
    monkeypatch.setattr(settings, "SURYA_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "VLLM_ENABLE_MTP", None)
    monkeypatch.setattr(settings, "SURYA_CONTAINER_RUNTIME", runtime_setting)
    monkeypatch.setattr(settings, "VLLM_GPUS", gpus)
    monkeypatch.setattr(vllm, "_checkpoint_has_mtp_head", lambda ckpt: has_mtp)

    def fake_which(name):
        if name in binaries or name.startswith("/"):
            base = name.rsplit("/", 1)[-1]
            return f"/usr/bin/{base}" if base in binaries else None
        return None

    monkeypatch.setattr(vllm.shutil, "which", fake_which)

    def fake_run(cmd, **kw):
        if cmd[1:] == ["--version"]:
            name = cmd[0].rsplit("/", 1)[-1]
            banner = version_banner or f"{name} version 1.0"
            return SimpleNamespace(returncode=0, stdout=banner, stderr="")
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vllm.subprocess, "run", fake_run)

    def fake_attach(
        backend,
        expected_model_name,
        spawn_fn,
        health_url_for,
        openai_url_for,
        startup_timeout,
        **kw,
    ):
        captured["handle"] = spawn_fn(8123)
        return SimpleNamespace(
            base_url="http://x/v1", model_name=expected_model_name, spawned_by_us=True
        )

    monkeypatch.setattr(vllm, "attach_or_spawn", fake_attach)
    vllm.VllmBackend().start()
    return captured["cmd"]


def _devices(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "--device"]


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


def test_docker_is_preferred_when_both_runtimes_present(monkeypatch):
    cmd = _argv(monkeypatch, binaries=("docker", "podman"))
    assert cmd[0] == "/usr/bin/docker"
    assert cmd[cmd.index("--runtime") + 1] == "nvidia"
    assert cmd[cmd.index("--gpus") + 1] == "device=0"
    assert "--device" not in cmd


def test_podman_is_used_when_docker_is_missing(monkeypatch):
    cmd = _argv(monkeypatch, binaries=("podman",))
    assert cmd[0] == "/usr/bin/podman"
    # podman has no `nvidia` OCI runtime: GPUs come in through CDI devices.
    assert "--runtime" not in cmd
    assert "--gpus" not in cmd
    assert _devices(cmd) == ["nvidia.com/gpu=0"]
    assert "--security-opt=label=disable" in cmd
    # Everything after the image is the vllm command line, unchanged.
    assert cmd[cmd.index(settings.VLLM_DOCKER_IMAGE) + 1] == "--model"


@pytest.mark.parametrize(
    "gpus,expected",
    [
        ("0,1", ["nvidia.com/gpu=0", "nvidia.com/gpu=1"]),
        ("all", ["nvidia.com/gpu=all"]),
        ("GPU-abc", ["nvidia.com/gpu=GPU-abc"]),
    ],
)
def test_podman_maps_vllm_gpus_to_cdi_devices(monkeypatch, gpus, expected):
    cmd = _argv(monkeypatch, binaries=("podman",), gpus=gpus)
    assert _devices(cmd) == expected


def test_explicit_runtime_setting_wins_over_auto(monkeypatch):
    cmd = _argv(monkeypatch, binaries=("docker", "podman"), runtime_setting="podman")
    assert cmd[0] == "/usr/bin/podman"
    assert _devices(cmd) == ["nvidia.com/gpu=0"]


def test_explicit_runtime_setting_accepts_a_path(monkeypatch):
    cmd = _argv(monkeypatch, binaries=("podman",), runtime_setting="/usr/bin/podman")
    assert cmd[0] == "/usr/bin/podman"
    assert "--runtime" not in cmd


def test_docker_shim_backed_by_podman_gets_podman_flags(monkeypatch):
    # podman-docker installs a `docker` binary that is really podman.
    cmd = _argv(
        monkeypatch, binaries=("docker",), version_banner="podman version 5.2.2"
    )
    assert cmd[0] == "/usr/bin/docker"
    assert "--runtime" not in cmd
    assert _devices(cmd) == ["nvidia.com/gpu=0"]


def test_missing_runtime_raises_spawn_error(monkeypatch):
    with pytest.raises(spawn.SpawnError, match="No container runtime found"):
        _argv(monkeypatch, binaries=())


def test_unknown_explicit_runtime_raises_spawn_error(monkeypatch):
    with pytest.raises(spawn.SpawnError, match="SURYA_CONTAINER_RUNTIME"):
        _argv(monkeypatch, binaries=("docker",), runtime_setting="nerdctl")


def test_runtime_kind_falls_back_to_binary_name(monkeypatch):
    def boom(cmd, **kw):
        raise OSError("cannot execute")

    monkeypatch.setattr(vllm.subprocess, "run", boom)
    assert vllm._runtime_kind("/opt/podman/bin/podman") == "podman"
    assert vllm._runtime_kind("/usr/bin/docker") == "docker"


def test_container_cleanup_and_logs_use_the_spawning_runtime(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="log line", stderr="")

    monkeypatch.setattr(spawn.subprocess, "run", fake_run)
    handle = spawn.SpawnHandle(
        pid=None,
        cleanup_id="surya-vllm-8123",
        cleanup_kind="docker",
        runtime="/usr/bin/podman",
    )
    assert spawn._capture_server_logs(handle, "vllm", tail=5) == "log line"
    spawn._stop_docker_container(handle.cleanup_id, handle.runtime)
    assert calls == [
        ["/usr/bin/podman", "logs", "--tail", "5", "surya-vllm-8123"],
        ["/usr/bin/podman", "stop", "surya-vllm-8123"],
    ]

    # Handles written before `runtime` existed keep going through docker.
    calls.clear()
    legacy = spawn.SpawnHandle(pid=None, cleanup_id="c", cleanup_kind="docker")
    spawn._stop_docker_container(legacy.cleanup_id, legacy.runtime)
    assert calls == [["docker", "stop", "c"]]
