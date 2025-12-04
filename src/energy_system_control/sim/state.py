from energy_system_control.sim.config import SimulationConfig
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class SimulationState:
    time: float = 0.0           # seconds
    time_id: int = 0
    time_vector: np.ndarray | None = None
    environmental_data: Dict[str, Any] = field(default_factory=dict)
    control_actions: Dict[str, Any] = field(default_factory=dict)
    time_step: float = 0.0

    def init_time_vector(self, cfg: SimulationConfig) -> None:
        self.time = cfg.time_start_h * 3600.0
        self.time_id = 0
        self.time_vector = np.arange(
            cfg.time_start_h * 3600.0,
            cfg.time_end_h * 3600.0,
            cfg.time_step_s,
        )
        self.time_step = cfg.time_step_s()