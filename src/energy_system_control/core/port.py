from typing import List, Dict
from energy_system_control.core.base_classes import InitContext

class Port():
    name: str
    connected_port: str
    flows: Dict[str, float]
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers
        self.connected_port = None
        self.reset_flow_data()  # Sets each 

    def reset_flow_data(self):
        self.flows = {name: None for name in self.layers}

    def reset_state_value(self):
        pass # Only implemented for selected port types
    
    def initialize(self, ctx: InitContext):
        pass

    def connect_port(self, port):
        # Connects the port to its connected port
        if type(self) != type(port):
            raise ValueError(f"Port types do not match: {self.name} has type {type(self)} while {port.name} has type {type(port)}")
        if self.connected_port is not None:
            raise ValueError(f"Port {self.name} is already connected to {self.connected_port.name}")
        self.connected_port = port

    def propagate_port_values(self):
        for layer in self.layers:
            if self.flows[layer] is not None and self.connected_port.flows[layer] is None:
                self.connected_port.flows[layer] = self.flows[layer]

    @staticmethod
    def create_port_of_type(port_name: str, port_type: str):
        match port_type:
            case 'heat':
                return HeatPort(port_name)
            case 'fluid':
                return FluidPort(port_name)
            case 'electricity':
                return ElectricPort(port_name)

class HeatPort(Port):
    T: float
    def __init__(self, name):
        super().__init__(name, ['heat'])

    def reset_state_value(self):
        self.T = None 

    def initialize(self):
        self.T = None


class FluidPort(Port):
    T: float
    def __init__(self, name):
        super().__init__(name, ['mass', 'heat'])
        self.T = None
        
    def reset_state_value(self):
        self.T = None

    def initialize(self):
        self.T = None

    def propagate_port_values(self):
        super().propagate_port_values()
        self.connected_port = self.T

class ElectricPort(Port):
    def __init__(self, name):
        super().__init__(name, ['electricity'])