from typing import Dict, List, Callable, Any, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np
from energy_system_control.core.ports import Port
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext
from energy_system_control.helpers import resample_with_interpolation

class Component:
    name: str
    time: float
    time_id: int
    time_step: float
    # environment: esc.Environment
    registry = {}
    ports_info: Dict[str, str]
    ports: Dict[str, Port]
    """Base class for components. Subclasses implement step(dt_s, nodes)."""
    def __init__(self, name: str, ports_info: Dict[str, str]):
        self.name = name
        self.ports_info = ports_info
        self.ports = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Component.registry[cls.__name__] = cls

    def attach(self, *, get_environmental_data: Callable[[str, Any], None]):
        self._environmental_data = get_environmental_data

    def create_ports(self):
        ports = {}
        for port_name, port_type in self.ports_info.items():
            self.ports[port_name] = Port.create_port_of_type(port_name, port_type)
            ports[port_name] = self.ports[port_name]
        return ports

    def set_inherited_fluid_port_values(self, state: SimulationState):
        return None, None
    
    def set_inherited_heat_port_values(self, state: SimulationState):
        return None, None
    
    def step(self):
        pass
    
    def initialize(self, ctx: InitContext):
        pass


@dataclass()
class TimeSeriesData:
    raw: pd.Series | pd.DataFrame
    var_type: Literal['energy', 'power', 'volume', 'mass', 'temperature']
    var_unit: Literal['Wh', 'kWh', 'MWh', 'W', 'kW', 'MW', 'l', 'm3', 'kg', 'C', 'K']
    data: np.ndarray | None = None
    energy_to_power_converter = {'Wh': 1e-3, 'kWh': 1.0, 'J': 1.0/3_600_000, 'kJ': 1.0/3600}

    def resample(self, time_step_h: float, sim_end_h: float):
        # Resamples the raw data to the format required 
        if self.raw is not None:
            target_freq = f"{int(time_step_h*3600)}s"
            if self.var_type == 'temperature':
                self.data = resample_with_interpolation(self.raw, target_freq, sim_end_h*3600.0, var_type="intensive")
            if self.var_type == 'power':
                self.data = resample_with_interpolation(self.raw, target_freq, sim_end_h*3600.0, var_type="intensive")
                if self.var_unit[0] != 'k':
                    self.data *= 1.0e-3
            elif self.var_type == 'energy':
                self.data = resample_with_interpolation(self.raw, target_freq, sim_end_h*3600.0, var_type="extensive")
                self.data = self.data * (1.0 / time_step_h) * self.energy_to_power_converter[self.var_unit]
            elif self.var_type in {'volume', 'mass'}:
                self.data = resample_with_interpolation(self.raw, target_freq, sim_end_h*3600.0, var_type="extensive")
            else:
                raise(ValueError, f'Unknown variable type {self.var_type}')
        else:
            raise(ValueError, 'No raw data available to resample for TimeSeriesDemand object')