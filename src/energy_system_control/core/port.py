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
        self.attribute_names = []
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
            if self.connected_port != port:
                raise ValueError(f"Port {self.name} is already connected to {self.connected_port.name}, cannot be connected to {port.name}")
            else:
                return
        self.connected_port = port

    def propagate_port_values(self):
        for layer in self.layers:
            if self.flows[layer] is not None and self.connected_port.flows[layer] is None:
                self.connected_port.flows[layer] = -self.flows[layer]
        for attribute_name in self.attribute_names:
            setattr(self.connected_port, attribute_name, getattr(self, attribute_name))

    @staticmethod
    def create_port_of_type(port_name: str, port_type: str):
        match port_type:
            case 'heat':
                return HeatPort(port_name)
            case 'fluid':
                return FluidPort(port_name)
            case 'electricity':
                return ElectricPort(port_name)
            case 'fuel':
                return FuelPort(port_name)
            case 'biogas':
                return BiogasPort(port_name)

class HeatPort(Port):
    T: float
    def __init__(self, name):
        super().__init__(name, ['heat'])

    def reset_state_value(self):
        self.T = None 

    def initialize(self, ctx: InitContext):
        self.T = None


class FluidPort(Port):
    T: float
    def __init__(self, name):
        super().__init__(name, ['mass', 'heat'])
        self.T = None
        self.attribute_names.append('T')
        
    def reset_state_value(self):
        self.T = None

    def initialize(self, ctx: InitContext):
        self.T = None


class ElectricPort(Port):
    def __init__(self, name):
        super().__init__(name, ['electricity'])


class FuelPort(Port):
    def __init__(self,name, other_properties: list = []):
        super().__init__(name, ['mass', 'chemical_energy']+other_properties)
        self.LHV=None
        self.attribute_names.append('LHV')


class BiogasPort(FuelPort):
    def __init__(self,name):
        super().__init__(name, ['volume'])
        self.methane_fraction=None 
        self.density=None
        self.attribute_names.append('methane_fraction')
        self.attribute_names.append('density')