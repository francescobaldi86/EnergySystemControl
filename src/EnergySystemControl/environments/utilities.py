from EnergySystemControl.environments.base_classes import Component, Node
from EnergySystemControl.environments.nodes import *
from EnergySystemControl.helpers import *
from typing import Dict, List


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

    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        # Assuming action is a value in kJ
        required_power = action / time_step
        if required_power > self.power_max:
            return {self.nodes[0]: self.power_max * time_step}
        elif required_power < self.power_min:
            return {self.nodes[0]: self.power_min * time_step}
        else:
            return {self.nodes[0]: action}



class HeatSource(Utility):
    def __init__(self, name: str, thermal_node: str, source_node: str, Qdot_max: float, efficiency: float):
        super().__init__(name, [thermal_node, source_node])
        self.Qdot_out = Qdot_max
        self.efficiency = efficiency

    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        return {self.nodes[0]: self.Qdot_out * action * time_step, 
                self.nodes[1]: -self.Qdot_out * action / self.efficiency * time_step}
    
class HeatPumpConstantEfficiency(HeatSource):
    COP: float
    def __init__(self, name: str, thermal_node: str, electrical_node: str, Qdot_max: float, COP = 3):
        super().__init__(name = name, thermal_node = thermal_node, source_node = electrical_node, Qdot_max = Qdot_max, efficiency = COP)
        self.COP = COP

    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        output = super().step(time_step, nodes, environmental_data, action)
        if action not in {0.0, 1.0}:
            raise OnOffComponent(f'The control input to the component {self.name} of type "HeatPumpConstantEfficiency" should be either 1 or 0. {action} was provided at time step {self.time}')
        return output


class BalancingUtility(Utility):
    # Balancing utilities are only used because their task is to ensure the balance of the nodes they are connected to. 
    def __init__(self, name: str, nodes: List[str]):
        super().__init__(name, nodes)

    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        output = {}
        for node in self.nodes:
            output[node] = -nodes[node].delta
        return output
    
class ColdWaterGrid(BalancingUtility):
    # This is a balancing note specifically for cold water. It balances both a thermal and a mass noed
    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        output = {}
        for node in self.nodes:
            if isinstance(nodes[node], MassNode):
                mass_node_name = node
            elif isinstance(nodes[node], ThermalNode):
                thermal_node_name = node
        output[mass_node_name] = -nodes[mass_node_name].delta
        output[thermal_node_name] = -nodes[mass_node_name].delta * 4.187 * environmental_data['Temperature cold water']  # Enthalpy content in kJ
        return output
    
    




class OnOffComponent(Exception):
    pass