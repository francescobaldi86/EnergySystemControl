from energy_system_control.controllers.base import HeaterControllerWithBandwidth
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import *

class HeatPumpRuleBasedController(HeaterControllerWithBandwidth):
    """
    Rule-based heat-pump controller that combines temperature and PV control.

    The controller first applies the bandwidth logic from
    :class:`HeaterControllerWithBandwidth` to keep the storage temperature near
    the comfort temperature. It then activates the heat pump whenever measured
    PV power is at least ``power_PV_activation`` and the storage temperature is
    below ``max_storage_temperature_for_activation``. This PV-based activation
    overrides the inherited bandwidth action, allowing available PV power to be
    used for heating while the storage remains below its safety limit.

    Parameters
    ----------
    name : str
        Name of the controller.
    controlled_component : str
        Name of the heat pump or heater controlled by this controller.
    temperature_sensor : str
        Name of the sensor measuring storage temperature.
    PV_power_sensor : str
        Name of the sensor measuring available PV power.
    temperature_comfort : float
        Target storage temperature in degrees Celsius.
    temperature_bandwidth : float
        Allowed temperature bandwidth above ``temperature_comfort``.
    power_PV_activation : float
        Minimum PV power required for PV-based heat-pump activation.
    max_storage_temperature_for_activation : float, default=60
        Maximum storage temperature for PV-based activation, in degrees
        Celsius.
    """
    def __init__(self, name, 
                 controlled_component: str, 
                 temperature_sensor: str, 
                 PV_power_sensor: str, 
                 temperature_comfort: float, 
                 temperature_bandwidth: float, 
                 power_PV_activation: float, 
                 max_storage_temperature_for_activation: float = 60,
                 minimum_time_off_between_activations_h: float | None = None,
                 minimum_time_on_between_deactivations_h: float | None = None):
        super().__init__(name, 
                         controlled_component, 
                         temperature_sensor, 
                         temperature_comfort, 
                         temperature_bandwidth,
                         minimum_time_off_between_activations_h,
                         minimum_time_on_between_deactivations_h)
        self.sensor_names.update({'PV power': PV_power_sensor})
        self.max_storage_temperature_for_activation = C2K(max_storage_temperature_for_activation)
        self.power_PV_activation = power_PV_activation
        self.PV_power_sensor_name = PV_power_sensor

    def _compute_action(self, state = SimulationState):
        # The principle of this controller is: 
        # - It tries to keep the temperature within limits, thus working as a "standard" bandwidth controller
        # - However, it also measures the power 
        power_PV = self.obs['PV power']
        if power_PV >= self.power_PV_activation and self.obs['Storage temperature'] < self.max_storage_temperature_for_activation:
            external_input = 1
        else:
            external_input = 0
        action = super().get_action(state, external_input)
        self.previous_action = action
        return action