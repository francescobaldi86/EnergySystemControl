import pandas as pd
import pytest

from energy_system_control.components.base import TimeSeriesData
from energy_system_control.core.base_classes import InitContext
from energy_system_control.sensors.data_sensors import DataSensor
from energy_system_control.sim.state import SimulationState


def make_time_series() -> TimeSeriesData:
    return TimeSeriesData(
        raw=pd.Series(
            [10.0, 20.0, 30.0],
            index=pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 02:00"]
            ),
        ),
        var_type="power",
        var_unit="kW",
        time_alignment="datetime",
    )


def test_data_sensor_reads_value_at_nearest_timestamp():
    sensor = DataSensor("test_sensor", make_time_series())
    state = SimulationState(
        time=80 * 60,
        simulation_start_datetime=pd.Timestamp("2025-01-01 00:00"),
    )

    assert sensor.measure(environment=None, state=state) == 20.0


def test_data_sensor_initialization_reads_initial_state():
    sensor = DataSensor("test_sensor", make_time_series())
    state = SimulationState(
        simulation_start_datetime=pd.Timestamp("2025-01-01 00:20")
    )

    sensor.initialize(InitContext(environment=None, state=state))

    assert sensor.get_measurement() == 10.0


def test_data_sensor_requires_simulation_start_datetime():
    sensor = DataSensor("test_sensor", make_time_series())

    with pytest.raises(ValueError, match="simulation_start_datetime"):
        sensor.measure(environment=None, state=SimulationState())
