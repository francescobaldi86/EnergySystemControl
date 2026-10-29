from typing import Dict, List, Callable, Any, Optional, Literal
from dataclasses import dataclass, fields
import pandas as pd
import numpy as np
from energy_system_control.helpers import TimeAlignment, resample_with_interpolation
from datetime import datetime

@dataclass(frozen=True, slots=True)
class InitContext:
    environment: Any
    state: Any
    rng: Optional[Any] = None
    logger: Optional[Any] = None
    config: Optional[dict] = None


@dataclass
class EnvironmentalData:
    temperature_ambient: float = 293.15  # K
    temperature_cold_water: float = 288.15  # K
    direct_irradiation: float = 0.0      # W/m^2
    diffuse_irradiation: float = 0.0     # W/m^2
    solar_zenith: float | None = None           # degrees
    solar_azimuth: float | None = None         # degrees       


@dataclass()
class TimeSeriesData:
    raw: pd.Series | pd.DataFrame
    var_type: Literal['energy', 'power', 'volume', 'mass', 'temperature', 'mass_flow', 'volume_flow', 'adimensional']
    var_unit: Literal['Wh', 'kWh', 'MWh', 'W', 'kW', 'MW', 'l', 'm3', 'kg', 'C', 'K']
    time_alignment: TimeAlignment
    data: np.ndarray | None = None
    energy_to_power_converter = {'Wh': 1e-3, 'kWh': 1.0, 'J': 1.0/3_600_000, 'kJ': 1.0/3600}

    def resample(self, 
                 time_step_h: float, 
                 simulation_end_h: float, 
                 simulation_start_datetime: datetime | None = None):
        # Resamples the raw data to the format required 
        if self.raw is not None:
            target_freq = f"{int(time_step_h*3600)}s"
            if self.var_type == 'temperature':
                self.data = resample_with_interpolation(df = self.raw, 
                                                        target_freq = target_freq,
                                                        simulation_end_s = simulation_end_h*3600.0,
                                                        simulation_start_datetime = simulation_start_datetime,
                                                        time_alignment = self.time_alignment,
                                                        var_type="intensive")
            if self.var_type == 'power':
                self.data = resample_with_interpolation(df = self.raw, 
                                                        target_freq = target_freq,
                                                        simulation_end_s = simulation_end_h*3600.0,
                                                        simulation_start_datetime = simulation_start_datetime,
                                                        time_alignment = self.time_alignment,
                                                        var_type="intensive")
                if self.var_unit[0] != 'k':
                    self.data = self.data * 1.0e-3
            elif self.var_type in {'mass_flow', 'volume_flow'}:
                self.data = resample_with_interpolation(self.raw, target_freq, simulation_end_h*3600.0, var_type="intensive")
            elif self.var_type == 'adimensional':
                self.data = resample_with_interpolation(self.raw, target_freq, simulation_end_h*3600.0, var_type="intensive")
            elif self.var_type == 'energy':
                self.data = resample_with_interpolation(df = self.raw, 
                                                        target_freq = target_freq,
                                                        simulation_end_s = simulation_end_h*3600.0,
                                                        simulation_start_datetime = simulation_start_datetime,
                                                        time_alignment = self.time_alignment,
                                                        var_type="extensive")
                self.data = self.data * (1.0 / time_step_h) * self.energy_to_power_converter[self.var_unit]
            elif self.var_type in {'volume', 'mass'}:
                self.data = resample_with_interpolation(df = self.raw, 
                                                        target_freq = target_freq,
                                                        simulation_end_s = simulation_end_h*3600.0,
                                                        simulation_start_datetime = simulation_start_datetime,
                                                        time_alignment = self.time_alignment,
                                                        var_type="extensive")
            else:
                raise ValueError(f'Unknown variable type {self.var_type}')
        else:
            raise(ValueError, 'No raw data available to resample for TimeSeriesDemand object')