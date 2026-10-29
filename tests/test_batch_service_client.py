"""Batch-service client: the request path put on the wire.

Torch-free and model-free - a stub HTTP server stands in for the real service,
so these run anywhere.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from surya.common.batch_service.client import BatchServiceClient
from surya.common.batch_service.config import ServiceConfig
from surya.inference.backends import spawn as spawn_mod


@pytest.fixture
def recording_server():
    """A stub batch server that records the path of every POST it receives.

    It answers any path so a wrong route shows up as a recorded value rather
    than as a 404 the client would surface as some other failure.
    """
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            paths.append(self.path)
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            body = json.dumps({"results": list(req.get("items", []))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], paths
    httpd.shutdown()
    httpd.server_close()


def _client(monkeypatch, port, base_suffix):
    """A client pinned to the stub server, with `attach_or_spawn` stubbed out."""
    base = f"http://127.0.0.1:{port}{base_suffix}"

    def fake_attach_or_spawn(**kwargs):
        return spawn_mod.SpawnedServer(
            base_url=base,
            health_url=f"http://127.0.0.1:{port}",
            model_name="stub",
            pid=None,
            backend="stub",
            spawned_by_us=False,
        )

    monkeypatch.setattr(
        "surya.common.batch_service.client.attach_or_spawn", fake_attach_or_spawn
    )
    config = ServiceConfig(
        backend="stub",
        model_name="stub",
        server_module="stub",
        host="127.0.0.1",
        external_url=base,
        port=port,
        autostart=False,
        startup_timeout=5.0,
        request_timeout=10.0,
        batch_wait_ms=0,
        max_batch=8,
    )
    return BatchServiceClient(
        config, encode_item=lambda i: i, decode_result=lambda r: r
    )


@pytest.mark.parametrize("base_suffix", ["/v1", "", "/v1/"])
def test_infer_posts_documented_route(monkeypatch, recording_server, base_suffix):
    """`infer` hits `/v1/infer` regardless of whether the resolved base URL
    already carries the OpenAI-style `/v1` prefix.

    The spawned base URL ends in `/v1` (see `BatchServiceClient._openai_url`),
    and httpx joins a leading-slash path onto the base path rather than
    replacing it, so a naive `post("/v1/infer")` goes out as `/v1/v1/infer`.
    """
    port, paths = recording_server
    client = _client(monkeypatch, port, base_suffix)

    assert client.infer([1, 2, 3]) == [1, 2, 3]
    assert paths == ["/v1/infer"]
