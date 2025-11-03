from EnergySystemControl.helpers import *
from typing import List, Dict

class Controller():
    name: str
    time: float
    time_id: int
    control_components: list
    sensors: list
    obs = dict
    def __init__(self, name, controlled_components: List[str], sensors: Dict[str, str]):
        self.name = name
        self.controlled_components = controlled_components
        self.sensors = sensors

    def get_obs(self, environment):
        self.obs = {var: environment.sensors[sensor_name].get_measurement(environment) for var, sensor_name in self.sensors.items()}

    def get_action(self):
        return None
    

class HeaterControllerWithBandwidth(Controller):
    """
    Controller for a heater with a bandwidth: it tries to keep the temperature within the specific band
    """
    def __init__(self, name, controlled_component: str, temperature_sensor: str, temperature_comfort: float, temperature_bandwidth: float):
        super().__init__(name, [controlled_component], {'Storage temperature': temperature_sensor})
        self.temperature_comfort = C2K(temperature_comfort)
        self.temperature_bandwidth = temperature_bandwidth
        self.previous_action = {controlled_component: 0}

    def get_action(self):
        temperature = self.obs["Storage temperature"]
        action = {}
        if temperature <= self.temperature_comfort:
            action[self.controlled_components[0]] = 1
        elif temperature <= self.temperature_comfort + self.temperature_bandwidth:
            action = self.previous_action
        else: 
            action[self.controlled_components[0]] = 0
        self.previous_action = action
        return action

class Inverter(Controller):
    def __init__(self, name, controlled_component: str, soc_sensor: str, grid_flow_sensor: str, SOC_min: float = 0.3, SOC_max: float = 0.9):
        super().__init__(name, [controlled_component], {'SOC': soc_sensor, 'balance': grid_flow_sensor})
        self.previous_action = {controlled_component: 0}
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max

    def get_action(self):
        # In the case of the inverter, the action is the energy required to the grid
        if self.obs['balance'] >= 0:
            if self.obs['SOC'] >= self.SOC_max:
                action = - self.obs['balance']
            else:
                action = 0
        else:
            if self.obs['SOC'] <= self.SOC_min:
                action = -self.obs['balance']
            else:
                action = 0
        return {self.controlled_components[0]: action}