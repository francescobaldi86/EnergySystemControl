from energy_system_control import Controller
from energy_system_control.helpers import *

class HeatPumpRuleBasedController(Controller):
    """
    Controller for a heater with a bandwidth: it tries to keep the temperature within the specific band
    """
    def __init__(self, name, control_node: str, controlled_component: str, temperature_comfort: float, temperature_bandwidth: float, power_PV_activation: float):
        super().__init__(name, [controlled_component], [control_node])
        self.temperature_comfort = C2K(temperature_comfort)
        self.temperature_bandwidth = temperature_bandwidth
        self.power_PV_activation = power_PV_activation
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