from energy_system_control.controllers.base import HeaterControllerWithBandwidth
from energy_system_control.helpers import *

class HeatPumpRuleBasedController(HeaterControllerWithBandwidth):
    """
    Controller for a heater with a bandwidth: it tries to keep the temperature within the specific band
    """
    def __init__(self, name, controlled_component: str, temperature_sensor: str, PV_power_sensor: str, temperature_comfort: float, temperature_bandwidth: float, power_PV_activation: float, max_storage_temperature_for_activation: float = 60):
        super().__init__(name, controlled_component, temperature_sensor, temperature_comfort, temperature_bandwidth)
        self.sensor_names.update({'PV power': PV_power_sensor})
        self.max_storage_temperature_for_activation = C2K(max_storage_temperature_for_activation)
        self.power_PV_activation = power_PV_activation
        self.PV_power_sensor_name = PV_power_sensor

    def get_action(self):
        # The principle of this controller is: 
        # - It tries to keep the temperature within limits, thus working as a "standard" bandwidth controller
        # - However, it also measures the power 
        action = super().get_action()
        power_PV = self.obs['PV power']
        if power_PV >= self.power_PV_activation and self.obs['Storage temperature'] < self.max_storage_temperature_for_activation:
            action[self.controlled_heater_name] = 1
        return action