from unittest.mock import Mock, patch

import httpx
import pytest

from surya.inference.backends.spawn import (
    SpawnError,
    SpawnHandle,
    attach_or_spawn,
    probe_model_ids,
)


def _probe_response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_probe_model_id_returns_all_model_ids():
    response = _probe_response(
        {
            "data": [
                {"id": "model-a"},
                {"id": "model-b"},
            ]
        }
    )

    with patch("surya.inference.backends.spawn.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = response

        assert probe_model_ids("http://server/v1") == ["model-a", "model-b"]


def test_probe_model_id_ignores_entries_without_valid_ids():
    response = _probe_response(
        {
            "data": [
                {"id": "model-a"},
                {},
                {"id": None},
                "not-a-model",
                {"id": ""},
                {"id": "model-b"},
            ]
        }
    )

    with patch("surya.inference.backends.spawn.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = response

        assert probe_model_ids("http://server/v1") == ["model-a", "model-b"]


def test_probe_model_id_returns_empty_list_on_failure():
    with patch("surya.inference.backends.spawn.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.side_effect = (
            httpx.ConnectError("connection failed", request=Mock())
        )

        assert probe_model_ids("http://server/v1") == []


def test_attach_accepts_expected_model_in_multiple_model_ids():
    with (
        patch("surya.inference.backends.spawn.probe_health", return_value=True),
        patch(
            "surya.inference.backends.spawn.probe_model_ids",
            return_value=["model-a", "expected-model", "model-b"],
        ),
    ):
        server = attach_or_spawn(
            backend="test",
            expected_model_name="expected-model",
            spawn_fn=Mock(),
            health_url_for=lambda port: f"http://server:{port}",
            openai_url_for=lambda port: f"http://server:{port}/v1",
            external_url="http://server/v1",
        )

    assert server.model_name == "expected-model"
    assert not server.spawned_by_us


def test_attach_rejects_when_expected_model_is_not_reported():
    with (
        patch("surya.inference.backends.spawn.probe_health", return_value=True),
        patch(
            "surya.inference.backends.spawn.probe_model_ids",
            return_value=["model-a", "model-b"],
        ),
        pytest.raises(SpawnError, match="expected-model"),
    ):
        attach_or_spawn(
            backend="test",
            expected_model_name="expected-model",
            spawn_fn=Mock(),
            health_url_for=lambda port: f"http://server:{port}",
            openai_url_for=lambda port: f"http://server:{port}/v1",
            external_url="http://server/v1",
        )


def test_fresh_server_uses_expected_model_from_multiple_model_ids(tmp_path):
    spawn_fn = Mock(
        return_value=SpawnHandle(
            pid=1234,
            cleanup_id="1234",
            cleanup_kind="process",
        )
    )

    with (
        patch(
            "surya.inference.backends.spawn._lock_path", return_value=tmp_path / "lock"
        ),
        patch("surya.inference.backends.spawn._read_sentinel", return_value=None),
        patch("surya.inference.backends.spawn._write_sentinel"),
        patch("surya.inference.backends.spawn.probe_health", return_value=False),
        patch("surya.inference.backends.spawn.wait_for_health", return_value=True),
        patch(
            "surya.inference.backends.spawn.probe_model_ids",
            return_value=["alias", "expected-model"],
        ),
    ):
        server = attach_or_spawn(
            backend="test",
            expected_model_name="expected-model",
            spawn_fn=spawn_fn,
            health_url_for=lambda port: f"http://server:{port}",
            openai_url_for=lambda port: f"http://server:{port}/v1",
            fixed_port=8765,
            keep_alive=True,
        )

    spawn_fn.assert_called_once_with(8765)
    assert server.model_name == "expected-model"
    assert server.spawned_by_us
