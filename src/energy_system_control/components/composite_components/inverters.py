from energy_system_control.components.base import CompositeComponent, ImplicitComponent
from energy_system_control.core.base_classes import InitContext
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import *
from energy_system_control.components.base import Bus
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
            self.converter = FixedEfficiencyInverterConverter(name = f'{name}_inverter', 
                                                     design_efficiency = design_efficiency)
        

    def get_internal_components(self):
        return {f'{self.name}_dc_bus': self.dc_bus, f'{self.name}_ac_bus': self.ac_bus, f'{self.name}_inverter': self.converter}
    
    def get_internal_connections(self):
        return [(f'{self.name}_dc_bus_internal_port', f'{self.name}_inverter_dc_port'),
               (f'{self.name}_ac_bus_internal_port', f'{self.name}_inverter_ac_port')]
    

class InverterConverter(ImplicitComponent):
    def __init__(self, name: str, design_efficiency: float):
        self.design_efficiency = design_efficiency
        self.ac_port_name = f'{name}_ac_port'
        self.dc_port_name = f'{name}_dc_port'
        super().__init__(name, {f'{name}_ac_port': 'electricity', f'{name}_dc_port': 'electricity'})
        

    def initialize(self, ctx: InitContext):
        pass

    def balance(self, state: SimulationState):
        ac_flow = self.ports[self.ac_port_name].flows['electricity']
        dc_flow = self.ports[self.dc_port_name].flows['electricity']
        if ac_flow is not None and dc_flow is not None:
            raise ValueError(f'Both ports of the inverter {self.name} have a flow value. This is not allowed.')
        elif ac_flow is None and dc_flow is None:
            return False, []
        elif ac_flow is not None:
            if ac_flow >= 0:  # Positive AC flow: the inverter is converting AC to DC
                self.ports[self.dc_port_name].flows['electricity'] = -ac_flow * self.efficiency
            elif ac_flow < 0:  # Negative AC flow: the inverter is converting DC to AC
                self.ports[self.dc_port_name].flows['electricity'] = -ac_flow / self.efficiency
            return True, [self.dc_port_name]
        elif dc_flow is not None:
            if self.ports[self.dc_port_name].flows['electricity'] >= 0:  # Positive DC flow: the inverter is converting DC to AC
                self.ports[self.ac_port_name].flows['electricity'] = -dc_flow * self.efficiency
            elif self.ports[self.dc_port_name].flows['electricity'] < 0:  # Negative DC flow: the inverter is converting AC to DC
                self.ports[self.ac_port_name].flows['electricity'] = -dc_flow / self.efficiency
            return True, [self.ac_port_name]
        else:
            raise ValueError('Something is wrong with the flows')
        
    @property
    def efficiency(self):
        return self.get_efficiency()

class FixedEfficiencyInverterConverter(InverterConverter):
    def get_efficiency(self):
        return self.design_efficiency

    