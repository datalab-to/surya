"""Server lifecycle: probe, filelock, sentinel, atexit cleanup.

Pattern: probe `/health` → if alive return handle → else acquire lock, re-probe,
spawn detached, write sentinel, register atexit kill (only the spawner cleans up).
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import httpx

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _cache_dir() -> Path:
    base = Path(os.path.expanduser("~/.cache/datalab/surya"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / ".write_probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return base
    except OSError:
        uid = getattr(os, "getuid", lambda: "user")()
        fallback = Path(tempfile.gettempdir()) / f"datalab-surya-{uid}"
        fallback.mkdir(parents=True, exist_ok=True, mode=0o700)
        return fallback


def _sentinel_path(backend: str) -> Path:
    return _cache_dir() / f"{backend}_server.json"


def _lock_path(backend: str) -> Path:
    return _cache_dir() / f"{backend}_server.lock"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def probe_health(base_url: str, timeout: float = 1.0) -> bool:
    """Returns True if the server reports healthy at /health."""
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_health(
    base_url: str,
    total_timeout: float = 300.0,
    interval: float = 1.0,
    is_alive: Optional[Callable[[], bool]] = None,
) -> bool:
    deadline = time.time() + total_timeout
    eperm_streak = 0
    while time.time() < deadline:
        try:
            if probe_health(base_url):
                return True
        except Exception:
            pass
        if is_alive is not None and not is_alive():
            return False
        # Detect sandboxed environments where outbound TCP sockets are blocked with EPERM
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                return True
            eperm_streak = 0
        except Exception as exc:
            if "Operation not permitted" in str(exc) or "[Errno 1]" in str(exc):
                eperm_streak += 1
                if eperm_streak >= 2:
                    logger.warning(
                        f"Outbound connection to {base_url} blocked by OS sandbox ({exc}); aborting health wait."
                    )
                    return False
            else:
                eperm_streak = 0
        time.sleep(interval)
    return False


def probe_model_ids(openai_base: str, timeout: float = 5.0) -> List[str]:
    """Returns all model ids reported by the running server, or [] on failure."""
    headers = {}
    if settings.SURYA_INFERENCE_API_KEY and settings.SURYA_INFERENCE_API_KEY != "EMPTY":
        headers["Authorization"] = f"Bearer {settings.SURYA_INFERENCE_API_KEY}"
    try:
        r = httpx.get(f"{openai_base}/models", timeout=timeout, headers=headers)
        r.raise_for_status()
        data = r.json()
        models = data.get("data") or []
        return [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
    except Exception:
        return []


def probe_model_id(openai_base: str, timeout: float = 5.0) -> Optional[str]:
    """Returns the first model id reported by the running server, or None on failure."""
    ids = probe_model_ids(openai_base, timeout=timeout)
    return ids[0] if ids else None


_DEFAULT_PROBE_MODEL_ID = probe_model_id


def _probed_models(openai_base: str, timeout: float = 5.0) -> List[str]:
    if probe_model_id is not _DEFAULT_PROBE_MODEL_ID:
        single = probe_model_id(openai_base)
        return [single] if single else []
    return probe_model_ids(openai_base, timeout=timeout)


def _match_expected_model(
    reported_models: List[str], expected_model_name: str
) -> Optional[str]:
    """Return the matching model id from reported_models if expected_model_name is served."""
    if not reported_models:
        return expected_model_name
    if expected_model_name in reported_models:
        return expected_model_name
    expected_tail = Path(expected_model_name).name
    for m in reported_models:
        if Path(m).name == expected_tail:
            return m
    return None


@dataclass
class SpawnedServer:
    base_url: str  # full openai base, e.g. "http://127.0.0.1:8765/v1"
    health_url: str  # base for /health, e.g. "http://127.0.0.1:8765"
    model_name: str  # what to pass as `model`
    pid: Optional[int]
    backend: str
    spawned_by_us: bool
    cleanup_fn: Optional[Callable[[], None]] = None


class SpawnError(RuntimeError):
    pass


def _read_sentinel(backend: str) -> Optional[dict]:
    p = _sentinel_path(backend)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_sentinel(backend: str, data: dict) -> None:
    _sentinel_path(backend).write_text(json.dumps(data))


def _delete_sentinel(backend: str) -> None:
    p = _sentinel_path(backend)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def _stop_process(pid: int, name: str) -> None:
    try:
        # Graceful first
        os.kill(pid, 15)  # SIGTERM
        for _ in range(20):
            try:
                os.kill(pid, 0)  # still alive?
            except ProcessLookupError:
                logger.info(f"Stopped {name} (pid {pid})")
                return
            time.sleep(0.5)
        # Hard
        os.kill(pid, 9)
        logger.warning(f"Force-killed {name} (pid {pid})")
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Failed to stop {name} (pid {pid}): {e}")


def _capture_server_logs(handle: "SpawnHandle", backend: str, tail: int = 100) -> str:
    """Best-effort tail of a server's logs, for surfacing startup failures."""
    try:
        if handle.cleanup_kind == "docker":
            runtime = handle.runtime or "docker"
            r = subprocess.run(
                [runtime, "logs", "--tail", str(tail), handle.cleanup_id],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return (r.stdout or "") + (r.stderr or "") or f"(no {runtime} logs)"
        log_path = _cache_dir() / f"{backend}_server.log"
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-tail:]) or "(empty log)"
    except Exception as e:
        return f"(could not capture logs: {e})"
    return "(no logs available)"


def _stop_docker_container(name: str, runtime: str = "docker") -> None:
    try:
        subprocess.run(
            [runtime, "stop", name], check=False, capture_output=True, timeout=30
        )
        logger.info(f"Stopped {runtime} container {name}")
    except Exception as e:
        logger.warning(f"Failed to stop {runtime} container {name}: {e}")


def _server_is_alive(handle: "SpawnHandle") -> bool:
    """True if the spawned server process or container is still running."""
    if handle.cleanup_kind == "process" and handle.pid:
        try:
            pid, status = os.waitpid(handle.pid, os.WNOHANG)
            if pid == handle.pid:
                return False
        except ChildProcessError:
            pass
        try:
            os.kill(handle.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    if handle.cleanup_kind == "docker" and handle.cleanup_id:
        runtime = handle.runtime or "docker"
        try:
            r = subprocess.run(
                [runtime, "inspect", "-f", "{{.State.Running}}", handle.cleanup_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode != 0:
                return False
            return r.stdout.strip().lower() == "true"
        except Exception:
            return True
    return True


_UNSET = object()


def attach_or_spawn(
    backend: str,
    expected_model_name: str,
    spawn_fn: Callable[[int], "SpawnHandle"],
    health_url_for: Callable[[int], str],
    openai_url_for: Callable[[int], str],
    startup_timeout: float = 600.0,
    *,
    external_url=_UNSET,
    autostart=_UNSET,
    fixed_port=_UNSET,
    keep_alive=_UNSET,
) -> SpawnedServer:
    """Generic attach-or-spawn with file lock and sentinel.

    `spawn_fn(port)` must launch the server detached and return a SpawnHandle
    with `pid` (int or None for docker) and a `cleanup_id` (e.g. container name).
    """
    if external_url is _UNSET:
        external_url = settings.SURYA_INFERENCE_URL
    if autostart is _UNSET:
        autostart = settings.SURYA_INFERENCE_AUTOSTART
    if fixed_port is _UNSET:
        fixed_port = settings.SURYA_INFERENCE_PORT
    if keep_alive is _UNSET:
        keep_alive = settings.SURYA_INFERENCE_KEEP_ALIVE

    # 0. If user pinned an external URL, attach without lock
    if external_url:
        base_url = external_url.rstrip("/")
        health_url = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
        if not probe_health(health_url):
            raise SpawnError(
                f"SURYA_INFERENCE_URL={base_url} is not reachable at /health. "
                "Start the server or unset the variable."
            )
        reported_models = _probed_models(base_url)
        matched_model = _match_expected_model(reported_models, expected_model_name)
        if matched_model is None:
            raise SpawnError(
                f"Model mismatch at {base_url}: expected {expected_model_name!r}, got {reported_models!r}. "
                "Stop the running server or unset SURYA_INFERENCE_URL."
            )
        return SpawnedServer(
            base_url=base_url,
            health_url=health_url,
            model_name=matched_model,
            pid=None,
            backend=backend,
            spawned_by_us=False,
        )

    # 1. Probe sentinel without lock (read-only fast path). This must NOT mutate
    # the sentinel: when many clients cold-start at once, one holds the lock
    # mid-spawn with its server still loading (so it reads unhealthy here). If an
    # unlocked waiter deleted the sentinel on that "unhealthy" read, it would then
    # acquire the lock, find no sentinel, and spawn a *second* server. So we only
    # attach here when healthy; otherwise fall through to the locked path, which
    # owns all sentinel deletion/replacement.
    existing = _read_sentinel(backend)
    if existing:
        port = existing.get("port")
        pid = existing.get("pid")
        if port and probe_health(health_url_for(port)):
            reported_models = _probed_models(openai_url_for(port))
            running_model = _match_expected_model(reported_models, expected_model_name)
            if running_model is None:
                raise SpawnError(
                    f"Existing {backend} server on port {port} serves {reported_models!r}, "
                    f"expected {expected_model_name!r}. Stop it before continuing."
                )
            logger.info(f"Attaching to existing {backend} server on port {port}")
            return SpawnedServer(
                base_url=openai_url_for(port),
                health_url=health_url_for(port),
                model_name=running_model,
                pid=pid,
                backend=backend,
                spawned_by_us=False,
            )

    if not autostart:
        raise SpawnError(
            f"No running {backend} server and autostart is disabled. "
            "Enable autostart or start the server manually."
        )

    # 2. Acquire filelock to prevent races
    try:
        from filelock import FileLock
    except ImportError as e:
        raise SpawnError(
            "filelock is required for server spawn. pip install filelock"
        ) from e

    lock = FileLock(str(_lock_path(backend)), timeout=120)
    with lock:
        # Re-check sentinel inside the lock
        existing = _read_sentinel(backend)
        if existing:
            port = existing.get("port")
            if port and probe_health(health_url_for(port)):
                reported_models = _probed_models(openai_url_for(port))
                running_model = _match_expected_model(
                    reported_models, expected_model_name
                )
                if running_model is None:
                    raise SpawnError(
                        f"Existing {backend} server on port {port} serves {reported_models!r}, "
                        f"expected {expected_model_name!r}."
                    )
                return SpawnedServer(
                    base_url=openai_url_for(port),
                    health_url=health_url_for(port),
                    model_name=running_model,
                    pid=existing.get("pid"),
                    backend=backend,
                    spawned_by_us=False,
                )

        # 3. Spawn fresh
        port = fixed_port or find_free_port()
        logger.info(f"Spawning {backend} server on port {port}")
        spawn_handle = spawn_fn(port)

        # 4. Write sentinel
        _write_sentinel(
            backend,
            {
                "port": port,
                "pid": spawn_handle.pid,
                "model": expected_model_name,
                "backend": backend,
                "cleanup_id": spawn_handle.cleanup_id,
                "cleanup_kind": spawn_handle.cleanup_kind,
                "runtime": spawn_handle.runtime,
            },
        )

        # 5. Register atexit cleanup (only spawner). Skipped when keep-alive is
        # set so the server outlives this process and later commands attach to
        # it via the sentinel. (_cleanup is still callable below on startup
        # failure, where we always tear a half-started server down.)
        cleaned_up = False

        def _cleanup():
            nonlocal cleaned_up
            if cleaned_up:
                return
            cleaned_up = True
            try:
                if spawn_handle.cleanup_kind == "docker":
                    _stop_docker_container(
                        spawn_handle.cleanup_id,
                        runtime=spawn_handle.runtime or "docker",
                    )
                elif spawn_handle.cleanup_kind == "process":
                    if spawn_handle.pid:
                        _stop_process(spawn_handle.pid, backend)
            finally:
                _delete_sentinel(backend)

        if keep_alive:
            logger.info(
                f"keep-alive: {backend} server on port {port} will stay up "
                f"after exit (cleanup_id={spawn_handle.cleanup_id!r})"
            )
        else:
            atexit.register(_cleanup)

        # 6. Wait for health
        health_url = health_url_for(port)
        if not wait_for_health(
            health_url,
            total_timeout=startup_timeout,
            is_alive=lambda: _server_is_alive(spawn_handle),
        ):
            # Grab the server's own logs *before* cleanup tears the (--rm)
            # container down, otherwise the actual failure reason is lost and
            # all the caller sees is this timeout.
            logs = _capture_server_logs(spawn_handle, backend)
            _cleanup()
            raise SpawnError(
                f"{backend} server failed to become healthy at {health_url} "
                f"within {startup_timeout}s.\n"
                f"--- last {backend} server logs ---\n{logs}"
            )

        # 7. Verify model name
        reported_models = _probed_models(openai_url_for(port))
        matched = _match_expected_model(reported_models, expected_model_name)
        if matched:
            expected_model_name = matched
        elif reported_models:
            logger.warning(
                f"{backend} server reports models={reported_models!r} "
                f"but expected {expected_model_name!r}; using {reported_models[0]!r}."
            )
            expected_model_name = reported_models[0]

        logger.info(
            f"{backend} server ready on port {port} (model={expected_model_name})"
        )
        return SpawnedServer(
            base_url=openai_url_for(port),
            health_url=health_url,
            model_name=expected_model_name,
            pid=spawn_handle.pid,
            backend=backend,
            spawned_by_us=True,
            cleanup_fn=None if keep_alive else _cleanup,
        )


@dataclass
class SpawnHandle:
    pid: Optional[int]
    cleanup_id: str  # container name for docker, str(pid) for process
    cleanup_kind: str  # "docker" | "process"
    runtime: Optional[str] = None  # "docker" | "podman" when cleanup_kind == "docker"
