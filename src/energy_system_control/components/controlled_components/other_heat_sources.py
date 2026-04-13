from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.sim.state import SimulationState

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