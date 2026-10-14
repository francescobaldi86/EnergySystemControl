from unittest.mock import Mock

import pytest

from energy_system_control.io.home_assistant import HomeAssistantClient


class TestHomeAssistantClient:
    def test_init_sets_url_without_trailing_slash_and_session_headers(self):
        client = HomeAssistantClient("https://example.com/", "test-token")

        assert client.url == "https://example.com"
        assert "Authorization" in client.session.headers
        assert client.session.headers["Content-Type"] == "application/json"

    def test_get_state_safe_returns_state_for_valid_response(self):
        client = HomeAssistantClient("https://example.com/", "test-token")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"state": "21.5"}
        client.session.get = Mock(return_value=response)

        result = client.get_state_safe("sensor.temperature")

        assert result == "21.5"
        client.session.get.assert_called_once_with(
            "https://example.com/api/states/sensor.temperature",
            timeout=5,
        )

    @pytest.mark.parametrize("state", ["unknown", "unavailable", None])
    def test_get_state_safe_returns_none_for_unusable_states(self, state):
        client = HomeAssistantClient("https://example.com/", "test-token")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"state": state} if state is not None else None
        client.session.get = Mock(return_value=response)

        assert client.get_state_safe("sensor.temperature") is None

    def test_get_float_state_converts_valid_state_to_float(self):
        client = HomeAssistantClient("https://example.com/", "test-token")
        client.get_state_safe = Mock(return_value="12.5")

        assert client.get_float_state("sensor.temperature") == 12.5

    @pytest.mark.parametrize(
        ("state", "default", "expected"),
        [
            ("not_a_number", 7.5, 7.5),
            (None, -1.0, -1.0),
        ],
    )
    def test_get_float_state_returns_default_for_invalid_or_missing_state(
        self, state, default, expected
    ):
        client = HomeAssistantClient("https://example.com/", "test-token")
        client.get_state_safe = Mock(return_value=state)

        assert client.get_float_state("sensor.temperature", default=default) == expected

    def test_call_service_posts_request_and_returns_json(self):
        client = HomeAssistantClient("https://example.com/", "test-token")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "ok"}
        client.session.post = Mock(return_value=response)

        result = client.call_service("light", "turn_on", "light.kitchen")

        assert result == {"status": "ok"}
        client.session.post.assert_called_once_with(
            "https://example.com/api/services/light/turn_on",
            json={"entity_id": "light.kitchen"},
            timeout=5,
        )
