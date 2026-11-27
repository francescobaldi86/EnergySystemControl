from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.port import FuelPort, BiogasPort
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
        heat_output = action
        if heat_output > self.max_power:
            raise ValueError(f'The required heat output is larger than the maximum output that the gas stove can provide')
        self.ports[self.heat_output_port_name].flows["heat"] = -heat_output
        power_input_port = self.ports[self.power_input_port_name]
        efficiency = self.get_efficiency()
        power_input_port.flows["electricity"] = heat_output / efficiency


class GasStove(HeatSource):

    def __init__(self, name: str, efficiency: float,
                 max_power: float, source_type: str = "fuel"):

        super().__init__(name, source_type)

        self.efficiency = efficiency
        self.max_power = max_power

    def get_heat_output(self, state):
        return self.max_power

    def get_efficiency(self, state=None):
        return self.efficiency

    def step(self, state, action):
        heat_output = action
        if heat_output > self.max_power:
            raise ValueError(f'The required heat output is larger than the maximum output that the gas stove can provide')
        self.ports[self.heat_output_port_name].flows["heat"] = -heat_output
        power_input_port = self.ports[self.power_input_port_name]
        efficiency = self.get_efficiency()

        # Potenza chimica richiesta al combustibile [kW]
        chemical_energy = heat_output / efficiency

        power_input_port.flows["chemical_energy"] = chemical_energy

        LHV = power_input_port.LHV
        density = power_input_port.density

        # Portata massica del biogas [kg/s], se LHV è in kJ/kg
        if LHV is not None and LHV > 0:
            power_input_port.flows["mass"] = chemical_energy / LHV
        else:
            power_input_port.flows["mass"] = 0.0

        # Portata volumetrica [m³/s]
        if density is not None and density > 0:
            power_input_port.flows["volume"] = (
                power_input_port.flows["mass"] / density
            )
        else:
            power_input_port.flows["volume"] = 0.0