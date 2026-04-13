from energy_system_control.components.base import ImplicitComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext


class _Inverter(ImplicitComponent):
    def __init__(self, name: str, design_efficiency: float, ac_bus_name: str, dc_bus_name: str):
        self.name = name
        self.design_efficiency = design_efficiency
        self.ac_port_name = ac_bus_name
        self.dc_port_name = dc_bus_name

    def initialize(self, ctx: InitContext):
        pass

    def balance(self, state: SimulationState):
        if self.ports[self.ac_port_name].flows['electricity'] is not None and self.ports[self.dc_port_name].flows['electricity'] is not None:
            raise ValueError(f'Both ports of the inverter {self.name} have a flow value. This is not allowed.')
        elif self.ports[self.ac_port_name].flows['electricity'] is None and self.ports[self.dc_port_name].flows['electricity'] is None:
            return False, []
        elif self.ports[self.ac_port_name].flows['electricity'] >= 0:  # Positive AC flow: the inverter is converting AC to DC
            self.ports[self.dc_port_name].flows['electricity'] = -self.ports[self.ac_port_name].flows['electricity'] * self.get_efficiency()
        elif self.ports[self.ac_port_name].flows['electricity'] < 0:  # Negative AC flow: the inverter is converting DC to AC
            self.ports[self.dc_port_name].flows['electricity'] = -self.ports[self.ac_port_name].flows['electricity'] / self.get_efficiency()
        elif self.ports[self.dc_port_name].flows['electricity'] >= 0:  # Positive DC flow: the inverter is converting DC to AC
            self.ports[self.ac_port_name].flows['electricity'] = -self.ports[self.dc_port_name].flows['electricity'] * self.get_efficiency()
        elif self.ports[self.dc_port_name].flows['electricity'] < 0:  # Negative DC flow: the inverter is converting AC to DC
            self.ports[self.ac_port_name].flows['electricity'] = -self.ports[self.dc_port_name].flows['electricity'] / self.get_efficiency()
        else:
            raise ValueError('Something is wrong with the flows')

class _FixedEfficiencyInverter(_Inverter):
    def get_efficiency(self):
        return self.design_efficiency