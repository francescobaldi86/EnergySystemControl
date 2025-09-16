from EnergySystemControl.helpers import *

class Controller():
    name: str
    time: float
    time_id: int
    def __init__(self, name):
        self.name = name

    def get_action(self):
        return None
    

class HeaterControllerWithBandwidth(Controller):
    """
    Controller for a heater with a bandwidth: it tries to keep the temperature within the specific band
    """
    def __init__(self, name, control_node: str, controlled_component: str, temperature_comfort: float, temperature_bandwidth: float, control_variable = 'T'):
        super().__init__(name)
        self.temperature_comfort = C2K(temperature_comfort)
        self.temperature_bandwidth = temperature_bandwidth
        self.control_node = control_node
        self.control_variable = control_variable
        self.controlled_component = controlled_component
        self.previous_action = 0

    def get_obs(self, environment):
        return getattr(environment.nodes[self.control_node], self.control_variable)

    def get_action(self, temperature_hp):
        if temperature_hp <= self.temperature_comfort:
            action =  1
        elif temperature_hp <= self.temperature_comfort + self.temperature_bandwidth:
            action = self.previous_action
        else: 
            action = 0
        self.previous_action = action
        return action