from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState
from abc import abstractmethod

class GenericControlledComponent(ControlledComponent):
    def __init__(self, name: str, port_type: str, max_power: float, type: str):
        self.port_name = f'{name}_{port_type}_port'
        self.port_type = port_type
        super().__init__(name, {self.port_name: port_type})
        match type:
            case 'source':
                self.power_min, self.power_max = 0.0, max_power
            case 'sink':
                self.power_min, self.power_max = -max_power, 0.0
            case 'bidirectional': 
                self.power_min, self.power_max = -max_power, max_power

    def step(self, state: SimulationState, action):
        # Assuming action is a value in kJ
        required_power = action / self.time_step
        # According to the port paradigm, flow is POSITIVE if ENTERING the component. 
        # On the other hand, required_power is positive when LEAVING the component. 
        if required_power > self.power_max: 
            self.ports[self.port_name].flows[self.port_type] = -self.power_max
        elif required_power < self.power_min: 
            self.ports[self.port_name].flows[self.port_type] = -self.power_min
        else:
            self.ports[self.port_name].flows[self.port_type] = -action


class HeatSource(ControlledComponent):
    # Partial class, implements a generic heat source
    heat_output_port_name: str
    power_input_port_name: str
    source_type: str
    def __init__(self, name: str, source_type: str):
        if source_type not in {'electricity', 'fuel'}:
            raise(KeyError, f'Source type for heat source {name} is not accepted. {source_type} was provided')
        self.heat_output_port_name = f'{name}_heat_output_port'
        self.power_input_port_name = f'{name}_{source_type}_input_port'
        self.source_type = source_type
        super().__init__(name, {self.heat_output_port_name: 'heat', self.power_input_port_name: source_type})

    @abstractmethod
    def get_heat_output(self, state: SimulationState):
        return NotImplementedError
    
    @abstractmethod
    def get_efficiency(self, state: SimulationState):
        raise NotImplementedError
    
    def get_power_input(self, state: SimulationState):
        return self.get_heat_output(state) / self.get_efficiency(state)

    def step(self, state: SimulationState, action):
        self.ports[self.heat_output_port_name].flows['heat'] = -self.get_heat_output(state) * action
        self.ports[self.power_input_port_name].flows[self.source_type] = self.get_power_input(state) * action