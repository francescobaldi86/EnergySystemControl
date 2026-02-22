# energy_system_control/sim/config.py
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class SimulationConfig:
    time_start_h: float = 0.0    # hours
    time_end_h: float = 8760.0   # hours
    time_step_h: float = 0.5     # hoursz
    simulation_start_datetime: pd.Timestamp | None = None
    prediction_horizon_margin_h: float = 25  # Represents how much more data we load to leave space for prediction

    @property
    def time_step_s(self) -> float:
        return self.time_step_h * 3600.0

    def time_end_s(self) -> float:
        return self.time_end_h * 3600.0
    
    def time_start_s(self) -> float:
        return self.time_start_h * 3600.0
