from energy_system_control.core.base_classes import Component, Node
from energy_system_control.core.nodes import MassNode, ThermalNode
from energy_system_control.helpers import *
from typing import Dict, List
from abc import abstractmethod


class Utility(Component):
    def __init__(self, name: str, nodes: List[str]):
        super().__init__(name, nodes)


class GenericUtility(Utility):
    def __init__(self, name: str, node:str, max_power: float, type: str):
        super().__init__(name, [node])
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
        if required_power > self.power_max:
            return {self.node_names[0]: self.power_max * self.time_step}
        elif required_power < self.power_min:
            return {self.node_names[0]: self.power_min * self.time_step}
        else:
            return {self.node_names[0]: action}


class HeatSource(Utility):
    # Partial class, implements a generic heat source
    Qdot_out: float
    def __init__(self, name: str, thermal_node: str, source_node: str):
        super().__init__(name, [thermal_node, source_node])

    @abstractmethod
    def get_heat_output(self):
        return NotImplementedError
    
    @abstractmethod
    def get_efficiency(self):
        raise NotImplementedError
    
    def get_power_input(self):
        return self.get_heat_output() / self.get_efficiency()

    def step(self, action):
        return {self.node_names[0]: self.get_heat_output() * action * self.time_step, 
                self.node_names[1]: -self.get_power_input() * action * self.time_step}


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
    def __init__(self, name: str, nodes: List[str]):
        super().__init__(name, nodes)

    def step(self, action):
        output = {}
        for node_name, node in self.nodes.items():
            output[node_name] = -node.delta
        return output


class ColdWaterGrid(BalancingUtility):
    # This is a balancing note specifically for cold water. It balances both a thermal and a mass noed
    def step(self, action):
        output = {}
        for node_name, node in self.nodes.items():
            if isinstance(node, MassNode):
                mass_node_name = node_name
            elif isinstance(node, ThermalNode):
                thermal_node_name = node_name
        output[mass_node_name] = -self.nodes[mass_node_name].delta
        output[thermal_node_name] = -self.nodes[mass_node_name].delta * 4.187 * self._environmental_data()['Temperature cold water']  # Enthalpy content in kJ
        return output