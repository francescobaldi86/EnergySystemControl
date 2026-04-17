from typing import Dict, List, Callable, Any, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np
from energy_system_control.core.port import Port
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext
from energy_system_control.helpers import resample_with_interpolation

class Component:
    name: str
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
        return {}
    
    def set_inherited_heat_port_values(self, state: SimulationState):
        return {}
    
    def step(self):
        pass
    
    def initialize(self, ctx: InitContext):
        pass


class ExplicitComponent(Component):
    # Definition of a component that does not depend on anything else to be simulated
    def does_something(self):
        pass

class ControlledComponent(Component):
    # Definition of a component that can be simulated fully after its related controller has been simulated
    def does_something(self):
        pass

class ImplicitComponent(Component):
    # Definition of a component that depends on other components to be simulated
    def balance(self, state: SimulationState):
        pass

class Bus(Component):
    def balance(self, state: SimulationState):
        """
        In the case of a bus, we assume that the sum of the flows into the bus is equal to the sum of the flows out of the bus.
        The balance function checks how many ports still haven't been assigned a value and calculates the missing one(s).
        
        Returns:
            tuple: (is_solved, updated_ports)
                - is_solved (bool): True if balance is complete, False if more than one port is missing
                - updated_ports (list): Names of ports whose flow was updated by this method
        """
        # Identify all unique layers across all ports
        all_layers = set()
        for port in self.ports.values():
            all_layers.update(port.layers)
        
        # For each layer, find ports with missing values
        ports_with_missing_per_layer = {}
        for layer in all_layers:
            ports_with_missing_per_layer[layer] = []
            for port_name, port in self.ports.items():
                if layer in port.layers and port.flows[layer] is None:
                    ports_with_missing_per_layer[layer].append(port_name)
        
        # Check if there are any layers with 2 or more missing ports
        for layer, missing_ports in ports_with_missing_per_layer.items():
            if len(missing_ports) >= 2:
                return False, []
        
        # At this point, each layer has at most 1 missing port
        # Check if all layers have exactly 1 missing port pointing to the same port(s)
        ports_to_update = set()
        for layer, missing_ports in ports_with_missing_per_layer.items():
            ports_to_update.update(missing_ports)
        
        # If no ports are missing, raise an error (already fully simulated)
        if len(ports_to_update) == 0:
            raise ValueError(f"Component '{self.name}' (Bus) is already fully simulated and should not be simulated again")
        
        # Calculate the missing flow value(s) for each port
        updated_port_names = []
        for port_name in ports_to_update:
            port = self.ports[port_name]
            for layer in port.layers:
                if port.flows[layer] is None:
                    # Calculate the sum of all other flows for this layer
                    sum_of_known_flows = 0.0
                    for other_port_name, other_port in self.ports.items():
                        if other_port_name != port_name and layer in other_port.layers:
                            if other_port.flows[layer] is not None:
                                sum_of_known_flows += other_port.flows[layer]
                    
                    # The missing flow is the negative of the sum
                    port.flows[layer] = -sum_of_known_flows
            
            updated_port_names.append(port_name)
        
        return True, updated_port_names


class Grid(Component):
    # Balancing utilities are only used because their task is to ensure the balance of the nodes they are connected to. 
    def __init__(self, name: str, utility_type):
        self.utility_type = utility_type
        self.port_name = f'{name}_{self.utility_type}_port'
        super().__init__(name, {self.port_name: self.utility_type})

    def step(self, state: SimulationState, action):
        pass  # In theory, nothing is needed here!


class StorageUnit(Component):
    """Generic storage unit"""
    max_capacity: float
    SOC: float

    def __init__(self, name: str, ports_info: dict):
        super().__init__(name, ports_info)

    def step(self, state: SimulationState, action):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        raise(NotImplementedError)
    
    def calculate_losses(self):
        raise(NotImplementedError)
    
    def check_storage_state(self):
        # This function must be implemented for each sub type
        raise(NotImplementedError)
    
    def initialize(self, state: SimulationState):
        self.SOC = self.SOC_0


class CompositeComponent(Component):
    """A composite component is a component that contains other components."""
    def get_internal_components(self):
        return {}
    
    def get_internal_connections(self):
        return[]


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
                self.data *= (1.0 / time_step_h) * self.energy_to_power_converter[self.var_unit]
            elif self.var_type in {'volume', 'mass'}:
                self.data = resample_with_interpolation(self.raw, target_freq, sim_end_h*3600.0, var_type="extensive")
            else:
                raise(ValueError, f'Unknown variable type {self.var_type}')
        else:
            raise(ValueError, 'No raw data available to resample for TimeSeriesDemand object')