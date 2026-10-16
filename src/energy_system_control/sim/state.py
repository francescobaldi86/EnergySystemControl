from energy_system_control.sim.config import SimulationConfig, RealTimeControlConfig
from energy_system_control.core.base_classes import EnvironmentalData
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta

@dataclass
class State:
    # Common temporal state
    time: float = 0.0             # Elapsed seconds from start
    time_id: int = 0              # Current step index
    time_step: float = 0.0        # Nominal step duration [s]
    start_datetime: datetime | None = None

    def initialize(self, cfg: SimulationConfig | RealTimeControlConfig):
        self.time = 0.0
        self.time_id = 0
        self.time_step = cfg.time_step_s

    @property
    def current_datetime(self) -> datetime | None:
        if self.start_datetime is None:
            return None

        return self.start_datetime + timedelta(seconds=self.time)

@dataclass
class SimulationState(State):
    time_vector: np.ndarray | None = None
    time_vector_for_prediction: np.ndarray | None = None  
    environmental_data: EnvironmentalData = field(default_factory=EnvironmentalData)
    control_actions: Dict[str, Any] = field(default_factory=dict)  
    
    def initialize(self, cfg: SimulationConfig):
        super().initialize(cfg)
        self._init_time_vector(cfg)
        self._init_environmental_data(cfg)

    def _init_time_vector(self, cfg: SimulationConfig) -> None:
        self.time_vector = np.arange(
            0.0,
            cfg.simulation_end_s,
            self.time_step,
        )
        self.simulation_start_datetime = cfg.simulation_start_datetime
        self.time_vector_for_prediction = np.arange(
            0.0,
            cfg.simulation_end_s + cfg.prediction_horizon_margin_h * 3600.0,
            self.time_step,
        )

    def _init_environmental_data(self, cfg: SimulationConfig):
        if cfg.environmental_defaults:
            # Make a copy to avoid modifying the same instance for multiple simulations
            self.environmental_data = deepcopy(cfg.environmental_defaults)
        else:
            self.environmental_data = EnvironmentalData()

@dataclass
class RealTimeControlState(State):
    actual_time: datetime = datetime.now().astimezone()
    scheduled_time_vector: np.ndarray | None = None  # Scheduled offsets from start_datetime [s]
    actual_time_vector: np.ndarray | None = None  # Actual execution times as Unix epoch seconds
    start_datetime: datetime = datetime.now().replace(second=0, microsecond=0)
    control_actions: Dict[str, Any] = field(default_factory=dict)

    def initialize(self, cfg: RealTimeControlConfig):
        super().initialize(cfg)
        self.scheduled_time_vector = np.arange(
                    0.0,
                    cfg.simulation_end_s,
                    cfg.time_step_s,
                )
        self.actual_time_vector = np.full(len(self.scheduled_time_vector), datetime(1900,1,1))