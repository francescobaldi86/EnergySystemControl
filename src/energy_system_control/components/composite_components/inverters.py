from energy_system_control.components.base import CompositeComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import *
from energy_system_control.components.base import Bus
from energy_system_control.components.implicit_components.converters import _Inverter, _FixedEfficiencyInverter
from typing import Dict, List



class Inverter(CompositeComponent):
    """
    The inverter is a component 
    """
    efficiency: float
    PV_port_name: str
    AC_output_port_name: str
    grid_port_name: str

    def __init__(self, name, design_efficiency: float = 0.92, efficiency_type: str = 'fixed'):
        super().__init__(name, {})
        self.PV_port_name = f'{name}_PV_input_port'
        self.AC_output_port_name = f'{name}_AC_output_port'
        self.grid_port_name = f'{name}_grid_input_port'
        self.ESS_port_name = f'{name}_ESS_port'
        self.dc_bus = Bus(name = f'{name}_dc_bus', 
                          ports_info = {self.PV_port_name: 'electricity', self.ESS_port_name: 'electricity', f'{self.name}_dc_bus_internal_port': 'electricity'})
        self.ac_bus = Bus(name = f'{name}_ac_bus', 
                          ports_info = {self.AC_output_port_name: 'electricity', self.grid_port_name: 'electricity', f'{self.name}_ac_bus_internal_port': 'electricity'})
        if efficiency_type == 'fixed':
            self.inverter = _FixedEfficiencyInverter(name = f'{name}_inverter', 
                                                     design_efficiency = design_efficiency)
        

    def get_internal_components(self):
        return {f'{self.name}_dc_bus': self.dc_bus, f'{self.name}_ac_bus': self.ac_bus, f'{self.name}_inverter': self.inverter}
    
    def get_internal_connections(self):
        return [(f'{self.name}_dc_bus_internal_port', f'{self.name}_inverter_dc_port'),
               (f'{self.name}_ac_bus_internal_port', f'{self.name}_inverter_ac_port')]
    

    