from energy_system_control.helpers import *
from typing import List, Dict
from energy_system_control.core.base_classes import Controller
    

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
            action[self.controlled_component_names[0]] = 1
        elif temperature <= self.temperature_comfort + self.temperature_bandwidth:
            action = self.previous_action
        else: 
            action[self.controlled_component_names[0]] = 0
        self.previous_action = action
        return action

class Inverter(Controller):
    def __init__(self, name, controlled_component: str, soc_sensor: str, grid_flow_sensor: str, SOC_min: float = 0.3, SOC_max: float = 0.9):
        super().__init__(name, [controlled_component], {'SOC': soc_sensor, 'balance': grid_flow_sensor})
        self.previous_action = {controlled_component: 0}
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max

    def get_action(self):
        # In the case of the inverter, the action is the energy required to balance the controlled sensor node (normally the exchange with the grid)
        # This involves two checks:
        #   - Power check (the power should not be higher than what is allowed by the battery)
        #   - Energy check (we should not be asking from the battery more energy then what is stored inside)
        if self.obs['balance'] >= 0:  # If the node balance is positive, the inverter will try to charge the battery
            energy_to_charge = min(self.controlled_components[self.controlled_component_names[0]].get_maximum_charge_power() * self.time_step, self.obs['balance'])  # First we limit based on battery power limits
            action = min(self.controlled_components[self.controlled_component_names[0]].max_battery_capacity * (self.SOC_max - self.controlled_components[self.controlled_component_names[0]].SOC), energy_to_charge)  # Then we limit based on the available energy
        else:
            energy_to_discharge = min(self.controlled_components[self.controlled_component_names[0]].get_maximum_discharge_power() * self.time_step, -self.obs['balance'])  # First we limit based on battery power limits
            action = -min(self.controlled_components[self.controlled_component_names[0]].max_battery_capacity * (self.controlled_components[self.controlled_component_names[0]].SOC - self.SOC_min), energy_to_discharge)  # Then we limit based on the available energy
        return {self.controlled_component_names[0]: action}