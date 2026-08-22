"""Client side of the batch service: attach-or-spawn + POST work over localhost.

Torch-free. Each model's predictor constructs a BatchServiceClient with its
ServiceConfig plus two hooks — `encode_item` (model input -> JSON unit) and
`decode_result` (JSON result -> model output). On first use the client attaches to
a running server or spawns one (file-locked, so N worker processes race safely and
only one server comes up).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

import httpx

from surya.common.batch_service.config import ServiceConfig
from surya.inference.backends.spawn import SpawnHandle, attach_or_spawn
from surya.logging import get_logger

logger = get_logger()


def _server_root(base_url: str) -> str:
    """Strip the OpenAI-style "/v1" suffix, leaving the server root."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


class BatchServiceClient:
    def __init__(
        self,
        config: ServiceConfig,
        encode_item: Callable[[Any], Any],
        decode_result: Callable[[Any], Any],
    ):
        self.config = config
        self._encode_item = encode_item
        self._decode_result = decode_result
        self._base_url: Optional[str] = None
        self._http: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    def _health_url(self, port: int) -> str:
        return f"http://{self.config.host}:{port}"

    def _openai_url(self, port: int) -> str:
        return f"http://{self.config.host}:{port}/v1"

    def _spawn_fn(self, port: int) -> SpawnHandle:
        cmd = [
            sys.executable,
            "-m",
            self.config.server_module,
            "--host",
            self.config.host,
            "--port",
            str(port),
            # Pin the checkpoint so a custom model_name matches what the server
            # reports (otherwise the spawned server loads the settings default
            # and attach_or_spawn's model-name check would reject it).
            "--checkpoint",
            self.config.model_name,
        ]
        logger.info(f"Spawning {self.config.backend} server: {' '.join(cmd)}")
        log_path = Path(
            f"~/.cache/datalab/surya/{self.config.backend}_server.log"
        ).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True
        )
        return SpawnHandle(
            pid=proc.pid, cleanup_id=str(proc.pid), cleanup_kind="process"
        )

    def _ensure_started(self) -> str:
        if self._base_url is not None:
            return self._base_url
        with self._lock:
            if self._base_url is not None:
                return self._base_url
            spawned = attach_or_spawn(
                backend=self.config.backend,
                expected_model_name=self.config.model_name,
                spawn_fn=self._spawn_fn,
                health_url_for=self._health_url,
                openai_url_for=self._openai_url,
                startup_timeout=self.config.startup_timeout,
                external_url=self.config.external_url,
                autostart=self.config.autostart,
                fixed_port=self.config.port,
                # Always keep_alive so spawn.py does NOT register an atexit kill:
                # many client processes share one server, and the spawning process
                # exiting must not tear it down mid-request for the others. The
                # server is persistent; we re-attach/re-spawn below if it goes away.
                keep_alive=True,
            )
            self._base_url = spawned.base_url
            self._http = httpx.Client(
                # `base_url` is the OpenAI-style endpoint, so it normally ends in
                # "/v1" (see `_openai_url`, and the documented form of the pinned
                # `<PREFIX>_SERVER_URL`). httpx *appends* a request path to the
                # base path instead of replacing it, so posting "/v1/infer"
                # against ".../v1" would go out as "/v1/v1/infer". Anchor the
                # client at the server root so the documented "/v1/infer" route
                # is what actually reaches the server.
                base_url=_server_root(self._base_url),
                # connect fails fast if the server is gone; reads can be long
                # (large batches). No keep-alive: avoids reusing a stale socket to
                # a server that has since died, which would hang instead of erroring.
                timeout=httpx.Timeout(self.config.request_timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=0),
            )
            return self._base_url

    def _reset(self) -> None:
        """Drop the cached server handle so the next call re-attaches/re-spawns."""
        with self._lock:
            if self._http is not None:
                try:
                    self._http.close()
                except Exception:
                    pass
            self._http = None
            self._base_url = None

    def infer(self, items: List[Any], params: Optional[dict] = None) -> List[Any]:
        if not items:
            return []
        payload = {
            "items": [self._encode_item(i) for i in items],
            "params": params or {},
        }
        # The server is persistent and shared; if it has gone away (crash, reboot,
        # manual kill, or a race during startup) the connection fails. Drop the
        # cached handle and re-attach/re-spawn, then retry once.
        last_err = None
        for attempt in range(2):
            self._ensure_started()
            try:
                resp = self._http.post("/v1/infer", json=payload)
                resp.raise_for_status()
                return [self._decode_result(r) for r in resp.json()["results"]]
            except (httpx.TransportError, httpx.TimeoutException) as e:
                # Any transport/timeout failure means the server is gone or
                # unreachable (crash, reboot, manual kill, half-open socket). Drop
                # the cached handle and re-attach/re-spawn, then retry once. HTTP
                # status errors (5xx from a live server) are NOT retried.
                last_err = e
                self._reset()
        raise last_err
