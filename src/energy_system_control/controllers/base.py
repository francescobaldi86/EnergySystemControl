from energy_system_control.helpers import *
from typing import List, Dict
from energy_system_control.components.base import Component
from energy_system_control.core.base_classes import Sensor

class Controller():
    name: str
    time: float
    time_id: int
    time_step: float
    controlled_component_names: List[str]
    sensor_names: Dict[str, str]
    controlled_components: Dict[str, Component]
    sensors: Dict[str, Sensor]
    obs: dict
    previous_action: dict
    def __init__(self, name, controlled_components: List[str], sensors: Dict[str, str]):
        """
        Class for a generic controller

        Parameters
        ----------
        name : str
            Name of the component
        controlled_components : list
            A list of the names of the controlled components
        sensors: dict
            A dictionary where each item corresponds to a sensor, and the respective key corresponds to the name of the variable read by the sensor 
        """
        self.name = name
        self.controlled_component_names = controlled_components
        self.sensor_names = sensors

    def get_obs(self, environment):
        self.obs = {var: sensor.get_measurement(environment) for var, sensor in self.sensors.items()}

    def load_controlled_components(self, components):
        self.controlled_components = {name: components[name] for name in self.controlled_component_names}
    
    def load_sensors(self, sensors):
        self.sensors = {var: sensors[sensor_name] for var, sensor_name in self.sensor_names.items()}

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
            action[self.controlled_component_names[0]] = 1
        elif temperature <= self.temperature_comfort + self.temperature_bandwidth:
            action = self.previous_action
        else: 
            action[self.controlled_component_names[0]] = 0
        self.previous_action = action
        return action

class InverterController(Controller):
    inverter_name: str
    battery_name: str
    def __init__(self, name, inverter_name: str, battery_name: str | None = None, SOC_min: float = 0.3, SOC_max: float = 0.9):
        self.inverter_name = inverter_name
        self.battery_name = battery_name
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.previous_action = {inverter_name: 0, battery_name: None} if battery_name else {inverter_name: 0}
        if self.battery_name:
            super().__init__(name, [inverter_name, battery_name], {})
        else:
            super().__init__(name, [inverter_name], {})
        

    def get_action(self):
        # In the case of the inverter, the action is the energy required to balance the controlled sensor node (normally the exchange with the grid)
        # This involves two checks:
        #   - Power check (the power should not be higher than what is allowed by the battery)
        #   - Energy check (we should not be asking from the battery more energy then what is stored inside)
        PV_power_input = self.controlled_components[self.inverter_name].ports[self.controlled_components[self.inverter_name].PV_port_name].flow['electricity']
        AC_output = self.controlled_components[self.inverter_name].ports[self.controlled_components[self.inverter_name].AC_output_port_name].flow['electricity']
        DC_output = AC_output / self.controlled_components[self.inverter_name].get_efficiency()
        DC_balance = PV_power_input + DC_output
        if self.battery_name:
            if DC_balance >= 0:  # If the node balance is positive, the inverter will try to charge the battery
                energy_to_charge = min(self.controlled_components[self.battery_name].get_maximum_charge_power() * self.time_step, DC_balance)  # First we limit based on battery power limits
                action = -min(self.controlled_components[self.battery_name].max_capacity * (self.SOC_max - self.controlled_components[self.battery_name].SOC), energy_to_charge)  # Then we limit based on the available energy
            else:
                energy_to_discharge = min(self.controlled_components[self.battery_name].get_maximum_discharge_power() * self.time_step, -DC_balance)  # First we limit based on battery power limits
                action = min(self.controlled_components[self.battery_name].max_capacity * (self.controlled_components[self.battery_name].SOC - self.SOC_min), energy_to_discharge)  # Then we limit based on the available energy
            return {self.inverter_name: action, self.battery_name: None}
        else:
            return {self.inverter_name: 0.0}