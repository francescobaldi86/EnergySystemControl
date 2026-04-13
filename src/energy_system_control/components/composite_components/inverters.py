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
    ESS_port_name: str
    grid_port_name: str

    def __init__(self, name, design_efficiency: float = 0.92, efficiency_type: str = 'fixed'):
        self.dc_bus = Bus(name = f'{name}_dc_bus', 
                          ports_info = {self.PV_port_name: 'electricity', self.ESS_port_name: 'electricity', f'{self.name}_dc_bus_port': 'electricity'})
        self.ac_bus = Bus(name = f'{name}_ac_bus', 
                          ports_info = {self.AC_output_port_name: 'electricity', self.grid_port_name: 'electricity', f'{self.name}_ac_bus_port': 'electricity'})
        if efficiency_type == 'fixed':
            self.inverter = _FixedEfficiencyInverter(name = f'{name}_inverter', design_efficiency = design_efficiency, ac_bus_name = self.ac_bus.name, dc_bus_name = self.dc_bus.name)
        super().__init__(name, {})

    