import numpy as np

from energy_system_control.sensors.sensors import Sensor
from energy_system_control.components.base import TimeSeriesData
from energy_system_control.core.base_classes import InitContext
from energy_system_control.core.base_environment import BaseEnvironment
from energy_system_control.sim.state import State

class DataSensor(Sensor):
    """Sensor that reads the time-series value nearest to the current time."""

    ts_data: TimeSeriesData

    def __init__(self, name: str, ts_data: TimeSeriesData):
        super().__init__(name)
        self.ts_data = ts_data

    def initialize(self, ctx: InitContext):
        self.ts_data.resample(
            time_step_h = ctx.state.time_step / 3600.0,
            simulation_end_h = ctx.state.scheduled_time_vector[-1] / 3600.0,
            simulation_start_datetime = ctx.state.start_datetime
        )
        self.ts_data.data[np.isnan(self.ts_data.data)] = 0.0
        super().initialize(ctx)

    def measure(self, environment: BaseEnvironment, state: State):
        self.current_measurement = self.ts_data.data[state.time_id]
        return self.current_measurement