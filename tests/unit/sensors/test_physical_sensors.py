from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from energy_system_control.core.base_classes import InitContext
from energy_system_control.sensors.physical_sensors import HomeAssistantSensor


class TestHomeAssistantSensor:
    def test_initialize_uses_home_assistant_client_from_environment(self):
        client = object()
        sensor = HomeAssistantSensor("temperature", "sensor.temperature")

        sensor.initialize(
            InitContext(
                environment=SimpleNamespace(clients={"home_assistant": client}),
                state=None,
            )
        )

        assert sensor.client is client

    def test_initialize_raises_value_error_if_client_missing(self):
        sensor = HomeAssistantSensor("temperature", "sensor.temperature")

        with pytest.raises(
            ValueError,
            match="Home Assistant client not found in the environment",
        ):
            sensor.initialize(
                InitContext(
                    environment=SimpleNamespace(clients={}),
                    state=None,
                )
            )

    def test_measure_float_reads_float_state_and_stores_value(self):
        client = Mock()
        client.get_float_state.return_value = 21.5
        sensor = HomeAssistantSensor("temperature", "sensor.temperature", is_float=True)

        sensor.initialize(
            InitContext(
                environment=SimpleNamespace(clients={"home_assistant": client}),
                state=None,
            )
        )
        result = sensor.measure(None, None)

        assert result == 21.5
        assert sensor.current_measurement == 21.5
        client.get_float_state.assert_called_once_with("sensor.temperature")

    def test_measure_non_float_reads_safe_state_and_stores_value(self):
        client = Mock()
        client.get_state_safe.return_value = "on"
        sensor = HomeAssistantSensor("status", "binary_sensor.front_door", is_float=False)

        sensor.initialize(
            InitContext(
                environment=SimpleNamespace(clients={"home_assistant": client}),
                state=None,
            )
        )
        result = sensor.measure(None, None)

        assert result == "on"
        assert sensor.current_measurement == "on"
        client.get_state_safe.assert_called_once_with("binary_sensor.front_door")
