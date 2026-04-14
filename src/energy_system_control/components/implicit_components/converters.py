from energy_system_control.components.base import ImplicitComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext


class _Inverter(ImplicitComponent):
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
                self.ports[self.dc_port_name].flows['electricity'] = -ac_flow * self.get_efficiency()
            elif ac_flow < 0:  # Negative AC flow: the inverter is converting DC to AC
                self.ports[self.dc_port_name].flows['electricity'] = -ac_flow / self.get_efficiency()
            return True, [self.dc_port_name]
        elif dc_flow is not None:
            if self.ports[self.dc_port_name].flows['electricity'] >= 0:  # Positive DC flow: the inverter is converting DC to AC
                self.ports[self.ac_port_name].flows['electricity'] = -dc_flow * self.get_efficiency()
            elif self.ports[self.dc_port_name].flows['electricity'] < 0:  # Negative DC flow: the inverter is converting AC to DC
                self.ports[self.ac_port_name].flows['electricity'] = -dc_flow / self.get_efficiency()
            return True, [self.ac_port_name]
        else:
            raise ValueError('Something is wrong with the flows')

class _FixedEfficiencyInverter(_Inverter):
    def get_efficiency(self):
        return self.design_efficiency