from typing import List, Dict
from energy_system_control.core.base_classes import Node

class Port(Node):
    connected_port: str
    def __init__(self, name, layers):
        super().__init__(name, layers)
        self.connected_port = None

    def reset_state_value(self):
        pass # Only implemented for selected port types

    def connect_port(self, port_name):
        self.connected_port = port_name

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
    def __init__(self, name):
        super().__init__(name, ['heat'])   


class FluidPort(Port):
    T: float
    def __init__(self, name):
        super().__init__(name, ['mass', 'heat'])
        
    def reset_state_value(self):
        self.T = None 

class ElectricPort(Port):
    def __init__(self, name):
        super().__init__(name, ['electricity'])