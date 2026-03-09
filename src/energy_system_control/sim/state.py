from energy_system_control.sim.config import SimulationConfig
from energy_system_control.core.base_classes import EnvironmentalData
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd
from copy import deepcopy

@dataclass
class SimulationState:
    time: float = 0.0           # seconds
    time_id: int = 0
    time_vector: np.ndarray | None = None
    simulation_start_datetime: pd.Timestamp | None = None
    environmental_data: EnvironmentalData = field(default_factory=EnvironmentalData)
    control_actions: Dict[str, Any] = field(default_factory=dict)
    time_step: float = 0.0
    
    def initialize(self, cfg: SimulationConfig):
        self._init_time_vector(cfg)
        self._init_environmental_data(cfg)

    def _init_time_vector(self, cfg: SimulationConfig) -> None:
        self.time = cfg.time_start_h * 3600.0
        self.time_id = 0
        self.time_vector = np.arange(
            cfg.time_start_h * 3600.0,
            cfg.time_end_h * 3600.0,
            cfg.time_step_s,
        )
        self.time_step = cfg.time_step_s
        self.simulation_start_datetime = cfg.simulation_start_datetime
        self.time_vector_for_prediction = np.arange(
            cfg.time_start_h * 3600.0,
            (cfg.time_end_h + cfg.prediction_horizon_margin_h) * 3600.0,
            cfg.time_step_s,
        )

    def _init_environmental_data(self, cfg: SimulationConfig):
        if cfg.environmental_defaults:
            # Make a copy to avoid modifying the same instance for multiple simulations
            self.environmental_data = deepcopy(cfg.environmental_defaults)
        else:
            self.environmental_data = EnvironmentalData()