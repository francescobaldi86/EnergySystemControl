# energy_system_control/sim/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationConfig:
    time_start_h: float = 0.0    # hours
    time_end_h: float = 8760.0   # hours
    time_step_h: float = 0.5     # hours

    @property
    def time_step_s(self) -> float:
        return self.time_step_h * 3600.0

    def time_end_s(self) -> float:
        return self.time_end_h * 3600.0
    
    def time_start_s(self) -> float:
        return self.time_start_h * 3600.0
