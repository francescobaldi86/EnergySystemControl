from energy_system_control.components.base import Component
from energy_system_control.helpers import *
from typing import Dict, List
from abc import abstractmethod


class Utility(Component):
    def __init__(self, name: str, nodes: List[str]):
        super().__init__(name, nodes)


class GenericUtility(Utility):
    def __init__(self, name: str, port_type: str, max_power: float, type: str):
        self.port_name = f'{name}_{port_type}_port'
        super().__init__(name, {self.port_name: port_type})
        match type:
            case 'source':
                self.power_min, self.power_max = 0.0, max_power
            case 'sink':
                self.power_min, self.power_max = -max_power, 0.0
            case 'bidirectional': 
                self.power_min, self.power_max = -max_power, max_power

    def step(self, action):
        # Assuming action is a value in kJ
        required_power = action / self.time_step
        # According to the port paradigm, flow is POSITIVE if ENTERING the component. 
        # On the other hand, required_power is positive when LEAVING the component. 
        if required_power > self.power_max: 
            self.ports[self.port_name].flow = -self.power_max * self.time_step
        elif required_power < self.power_min: 
            self.ports[self.port_name].flow = -self.power_min * self.time_step
        else:
            self.ports[self.port_name].flow = -action


class HeatSource(Component):
    # Partial class, implements a generic heat source
    Qdot_out: float
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
    def get_heat_output(self):
        return NotImplementedError
    
    @abstractmethod
    def get_efficiency(self):
        raise NotImplementedError
    
    def get_power_input(self):
        return self.get_heat_output() / self.get_efficiency()

    def step(self, action):
        self.ports[self.heat_output_port_name].flow['heat'] = -self.get_heat_output() * action * self.time_step
        self.ports[self.power_input_port_name].flow[self.source_type] = self.get_power_input() * action * self.time_step


class SimplifiedHeatSource(HeatSource):
    def __init__(self, name: str, thermal_node: str, source_node: str, Qdot_max: float, efficiency: float):
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
        super().__init__(name = name, thermal_node = thermal_node, source_node = source_node)
        self.Qdot_out = Qdot_max
        self.efficiency = efficiency

    def get_heat_output(self):
        return self.Qdot_out
    
    def get_efficiency(self):
        return self.efficiency


class BalancingUtility(Utility):
    # Balancing utilities are only used because their task is to ensure the balance of the nodes they are connected to. 
    def __init__(self, name: str, utility_type):
        self.utility_type = utility_type
        self.port_name = f'{name}_{self.utility_type}_port'
        super().__init__(name, {self.port_name: self.utility_type})

    def step(self, action):
        pass  # In theory, nothing is needed here!


class ColdWaterGrid(BalancingUtility):
    # Specific balancing utility for the cold water grid. Useful because it reads the temperature of the water
    def set_inherited_fluid_port_values(self):
        self.ports[self.port_name].T = self._environmental_data()['Temperature cold water']  # Enthalpy content in kJ
        return self.port_name, self._environmental_data()['Temperature cold water']
        

class Inverter(Component):
    """
    The inverter is a component 
    """
    efficiency: float
    PV_port_name: str
    AC_output_port_name: str
    ESS_port_name: str
    grid_port_name: str
    def __init__(self, name, efficiency: float = 0.92):
        self.PV_port_name = f'{name}_PV_input_port'
        self.AC_output_port_name = f'{name}_output_port'
        self.ESS_port_name = f'{name}_ESS_port'
        self.grid_port_name = f'{name}_grid_input_port'
        self.efficiency = efficiency
        ports_info = {
            self.PV_port_name: 'electricity',
            self.AC_output_port_name: 'electricity',
            self.ESS_port_name: 'electricity',
            self.grid_port_name: 'electricity'
        }
        super().__init__(name, ports_info)

    def get_efficiency(self):
        return self.efficiency

    def step(self, action: float = 0.0):
        # The action is the power exchanged with the battery. The AC grid input is calculated
        ESS_power = action
        # If there is no battery, there is no controller and the action is simply balancing produced PV with request
        # AC input is calculated based on the energy balance:
        PV_power_input = self.ports[self.PV_port_name].flow['electricity']
        AC_output = self.ports[self.AC_output_port_name].flow['electricity']
        AC_input = (PV_power_input + ESS_power) * self.get_efficiency() + AC_output
        self.ports[self.grid_port_name].flow['electricity'] = -AC_input
        self.ports[self.ESS_port_name].flow['electricity'] = action