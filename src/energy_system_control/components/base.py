from typing import Dict, List, Callable, Any
from energy_system_control.core.base_classes import Node
from energy_system_control.core.ports import Port
from energy_system_control.sim.state import SimulationState

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
    
    def initialize(self, state: SimulationState | None = None):
        pass