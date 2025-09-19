from EnergySystemControl.helpers import *
from typing import List

class Controller():
    name: str
    time: float
    time_id: int
    control_components: list
    control_nodes: list
    def __init__(self, name, controlled_components: List[str], control_nodes: List[str]):
        self.name = name
        self.controlled_components = controlled_components
        self.control_nodes = control_nodes

    def get_action(self):
        return None
    

class HeaterControllerWithBandwidth(Controller):
    """
    Controller for a heater with a bandwidth: it tries to keep the temperature within the specific band
    """
    def __init__(self, name, control_node: str, controlled_component: str, temperature_comfort: float, temperature_bandwidth: float):
        super().__init__(name, [controlled_component], [control_node])
        self.temperature_comfort = C2K(temperature_comfort)
        self.temperature_bandwidth = temperature_bandwidth
        self.previous_action = {controlled_component: 0}

    def get_obs(self, environment):
        return environment.nodes[self.control_nodes[0]].T

    def get_action(self, temperature_hp):
        action = {}
        if temperature_hp <= self.temperature_comfort:
            action[self.controlled_components[0]] = 1
        elif temperature_hp <= self.temperature_comfort + self.temperature_bandwidth:
            action = self.previous_action
        else: 
            action[self.controlled_components[0]] = 0
        self.previous_action = action
        return action

class Inverter(Controller):
    def __init__(self, name, control_node: str, controlled_component: str):
        super().__init__(name, [controlled_component], [control_node])
        self.previous_action = {controlled_component: 0}

    def get_obs(self, environment):
        return {'SOC': environment.nodes[self.control_nodes[0]].SOC,
                'balance': environment.nodes[self.control_nodes[0]].delta,
                'battery capacity': environment.nodes[self.control_nodes[0]].max_capacity}

    def get_action(self, obs):
        # In the case of the inverter, the action is the energy required to the grid
        if obs['balance'] >= 0:
            available_batt_space = obs['battery capacity'] * (1 - obs['SOC'])
            if available_batt_space > obs['balance']:
                action = 0
            else:
                action = available_batt_space - obs['balance']
        else:
            battery_energy_level = obs['battery capacity'] * obs['SOC']
            if battery_energy_level >= -obs['balance']: 
                action = 0
            else:
                action = -(obs['balance'] + battery_energy_level)
        return {self.controlled_components[0]: action}