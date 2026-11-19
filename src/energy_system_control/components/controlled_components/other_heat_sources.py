from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.sim.state import SimulationState
from energy_system_control.components.controlled_components.base import ControlledComponent

class ResistanceHeater(HeatSource):
    def __init__(self, name: str, Qdot_max: float, efficiency: float):
        """
        Model of heat pump based on a generic heat source with fixed heat output and fixed efficiency

        Parameters
        ----------
        name : str
            Name of the component
        thermal_node : str
         	Name of the node for the thermal connection. Most times, it is the thermal node of the storage tank
        source_node : str
           	Name of the node for the input connection (fuel, electricity)
        Qdot_max : float
            Output heat flow [kW] of the unit
        efficiency : float
            Efficiency [-] of the unit
        """
        super().__init__(name = name, source_type='electricity')
        self.Qdot_out = Qdot_max
        self.efficiency = efficiency

    def get_heat_output(self, state: SimulationState):
        return self.Qdot_out
    
    def get_efficiency(self, state: SimulationState):
        return self.efficiency

'''class GasBoiler(HeatSource):

    def __init__(self, name: str, efficiency: float, max_power: float, source_type: str = "biogas"):
        super().__init__(name, source_type)
        self.efficiency = efficiency    # Rendimento (0.0 - 1.0)
        self.max_power = max_power      # Potenza termica massima [kW]

    def get_heat_output(self, state):
        if self.ports[self.power_input_port_name].LHV > 0.0:
            return self.max_power
        else:
            return 0.0


    def get_efficiency(self, state=None) -> float:
        return self.efficiency'''

class GasBoiler(HeatSource):

    def __init__(self, name: str, efficiency: float,
                 max_power: float, source_type: str = "biogas"):

        super().__init__(name, source_type)

        self.efficiency = efficiency
        self.max_power = max_power

    def get_heat_output(self, state):
        return self.max_power
    
    def get_efficiency(self, state=None):
        return self.efficiency

class InductionStove(HeatSource):

    def __init__(self, name: str, efficiency: float,
                 max_power: float, source_type: str = "electricity"):

        super().__init__(name, source_type)

        self.efficiency = efficiency
        self.max_power = max_power

    def get_heat_output(self, state):
        return self.max_power
    
    def get_efficiency(self, state=None):
        return self.efficiency

    def step(self, state, action):
        if action is None:
            required_power = -self.ports[f'{self.name}_heat_output_port'].flows['heat']
            if required_power <= self.max_power:
                self.ports[self.power_input_port_name].flows["electricity"] = required_power / self.efficiency
            else:
                raise ValueError(f'Required power {required_power:.2f} is higher than the maximum power of the stove {self.max_power:.2f}')