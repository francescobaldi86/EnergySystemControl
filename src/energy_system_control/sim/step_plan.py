"""Compiled execution plan for a simulation run.

`Environment.build_step_plan()` builds a `StepPlan` once at the start of
`Simulator.run()`. It precomputes every lookup the per-timestep hot loop
would otherwise repeat (port -> result-column mappings, propagation pairs,
balance-check pairs, controller/sensor save plans, solar angles for the whole
horizon), so that `Simulator._step` runs as tight loops over plain lists.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StepPlan:
    # Objects iterated every step, in the exact order the old code used.
    ports: List[Any] = field(default_factory=list)            # reset each step
    sensors: List[Any] = field(default_factory=list)          # measured each step
    controllers: List[Any] = field(default_factory=list)      # in user-supplied order
    inherit: List[Any] = field(default_factory=list)          # components for set_inherited_fluid/heat_port_values
    explicit: List[Any] = field(default_factory=list)         # simulated 1st
    controlled: List[Any] = field(default_factory=list)       # simulated 2nd
    algebraic: List[Any] = field(default_factory=list)        # Bus + ImplicitComponent loop
    storage: List[Any] = field(default_factory=list)          # simulated after the loop
    grid: List[Any] = field(default_factory=list)             # simulated last

    # Per component: [(port, connected_port, layers, copy_T)] used by
    # _take_component_step to mirror flows (and temperature when copy_T)
    # onto connected ports.
    step_propagation: Dict[str, List[Tuple[Any, Any, List[str], bool]]] = field(default_factory=dict)

    # Result saving: entries aligned with the columns of SimulationData arrays
    # (registry column i corresponds to the i-th entry).
    port_save: List[Tuple[Any, str]] = field(default_factory=list)        # (port, layer); layer 'temperature' reads port.T
    controller_save: List[Tuple[Any, Dict[str, int], Optional[int], Optional[int]]] = field(default_factory=list)
    sensor_save: List[Any] = field(default_factory=list)                  # sensors in column order

    # Balance check: [(port_a, port_b, layer)] for every connection x layer.
    balance_checks: List[Tuple[Any, Any, str]] = field(default_factory=list)

    # Environmental data: one datetime per step, and solar angles precomputed
    # for the whole horizon (None when latitude/longitude are not set).
    datetimes: Optional[pd.DatetimeIndex] = None
    env_field_names: List[str] = field(default_factory=list)
    solar_zenith: Optional[np.ndarray] = None
    solar_azimuth: Optional[np.ndarray] = None
