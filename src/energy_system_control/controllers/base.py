from typing import List, Dict, Any
from abc import ABC, abstractmethod
from energy_system_control.helpers import *
from energy_system_control.components.base import Component
from energy_system_control.core.base_classes import InitContext
from energy_system_control.sensors.sensors import Sensor
from energy_system_control.sim.state import SimulationState
from energy_system_control.controllers.predictors import Predictor

class Controller(ABC):
    name: str
    time: float
    time_id: int
    time_step: float
    controlled_component_names: List[str]
    sensor_names: Dict[str, str]
    predictor_names: Dict[str, str]
    controlled_components: Dict[str, Component]
    sensors: Dict[str, Sensor]
    predictors: Dict[str, Predictor]
    obs: dict
    previous_action: dict
    def __init__(self, 
                 name, 
                 controlled_components: List[str], 
                 sensors: Dict[str, str],
                 minimum_time_off_between_activations_h: dict = {},
                 minimum_time_on_between_deactivations_h: dict = {},
                 predictors: Dict[str, str] = {}):
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
        minimum_time_off_between_activations_h: dict, optional
            Minimum time that the controller must wait before turning on again the component [hours]. Defaults to an empty dictionary
        minimum_time_on_between_deactivations_h: dict, optional
            Minimum time that the controller must wait before turning off again the component [hours]. Defaults to an empty dictionary
        predictors: dict, optional
            A dictionary of the predictors used by the controller
        """
        self.name = name
        self.controlled_component_names = controlled_components
        self.sensor_names = sensors
        self.predictor_names = predictors
        self.minimum_time_on_between_activations = {k: v*3600 for k, v in minimum_time_off_between_activations_h.items()}
        self.minimum_time_off_between_deactivations = {k: v*3600 for k, v in minimum_time_on_between_deactivations_h.items()}
        self.on_off_time_limitations = bool(len(self.minimum_time_off_between_deactivations) + len(self.minimum_time_on_between_activations))

    def initialize(self, ctx: InitContext):
        self.load_controlled_components(ctx.environment.components)
        self.load_sensors(ctx.environment.sensors)
        self.load_predictors(ctx.environment.predictors)
        self.previous_action = {comp: 0 for comp in self.controlled_components}
        self.time_since_last_state_change = {comp: 0.0 for comp in self.controlled_components}

    def get_obs(self, environment, state) -> Dict[str, Any]:
        self.obs = {var: sensor.get_measurement() for var, sensor in self.sensors.items()}
        self.predictions = {var: predictor.predict(self.horizon, state) for var, predictor in self.predictors.items()}
        return self.obs

    def load_controlled_components(self, components: Dict[str, Any]):
        self.controlled_components = {name: components[name] for name in self.controlled_component_names}
    
    def load_sensors(self, sensors: Dict[str, Any]):
        self.sensors = {var: sensors[sensor_name] for var, sensor_name in self.sensor_names.items()}
    
    def load_predictors(self, predictors: Dict[str, Any]):
        self.predictors = {var: predictors[predictor_name] for var, predictor_name in self.predictor_names.items()}

    def get_action(self, state, **kwargs):
        action = self._compute_action(state, **kwargs)
        action = self.check_dynamic_limitations(state, action)
        self.previous_action = action
        return action

    @abstractmethod
    def _compute_action(self, state, **kwargs):
        raise NotImplementedError

    def check_dynamic_limitations(self, state, action):
        action = self.check_time_elapsed_since_last_state_change(
            state, action
        )
        # Future:
        # action = self.check_ramp_limits(state, action)
        return action

    def check_time_elapsed_since_last_state_change(self, state, action):
        # Method to ensure that the controlled component is not turned ON/OFF too often
        if self.on_off_time_limitations:
            for component_name in self.controlled_component_names:
                # Check whether the action is valid
                # First, if no change of action is required, all is good and we simply update the time elapsed since last state change
                if action[component_name] == self.previous_action[component_name]:
                    self.time_since_last_state_change[component_name] += state.time_step
                elif action[component_name] == True and self.previous_action[component_name] == False:
                    if self.time_since_last_state_change[component_name] >= self.minimum_time_off_between_deactivations[component_name]:
                        self.time_since_last_state_change[component_name] = 0.0
                    else:
                        action[component_name] = False
                        self.time_since_last_state_change[component_name] += state.time_step
                elif action[component_name] == False and self.previous_action[component_name] == True:
                    if self.time_since_last_state_change[component_name] >= self.minimum_time_on_between_activations[component_name]:
                        self.time_since_last_state_change[component_name] = 0.0
                    else:
                        action[component_name] = True
                        self.time_since_last_state_change[component_name] += state.time_step
                else:
                    raise ValueError('There should be no other option')

        return action
    
    

class HeaterControllerWithBandwidth(Controller):
    """
    Controller for a heater with a bandwidth: it tries to keep the temperature within the specific band
    """
    def __init__(self, 
                 name, 
                 controlled_component: str, 
                 temperature_sensor: str,
                 temperature_comfort: float, 
                 temperature_bandwidth: float,
                 minimum_time_off_between_activations_h: float | None = None,
                 minimum_time_on_between_deactivations_h: float | None = None):
        if minimum_time_off_between_activations_h is not None:
            minimum_time_off_between_activations_h = {controlled_component: minimum_time_off_between_activations_h}
        else:
            minimum_time_off_between_activations_h = {}
        if minimum_time_on_between_deactivations_h is not None:
            minimum_time_on_between_deactivations_h = {controlled_component: minimum_time_on_between_deactivations_h}
        else:
            minimum_time_on_between_deactivations_h = {}
        super().__init__(name, 
                         [controlled_component], 
                         {'Storage temperature': temperature_sensor}, 
                         minimum_time_off_between_activations_h=minimum_time_off_between_activations_h,
                         minimum_time_on_between_deactivations_h=minimum_time_on_between_deactivations_h)
        self.temperature_comfort = C2K(temperature_comfort)
        self.temperature_bandwidth = temperature_bandwidth
        self.temperature_sensor_name = temperature_sensor
        self.controlled_heater_name = controlled_component

    def _compute_action(self, state: SimulationState, external_input: int | float = 0):
        temperature = self.obs["Storage temperature"]
        action = {}
        if temperature <= self.temperature_comfort:
            action[self.controlled_heater_name] = 1
        elif temperature <= self.temperature_comfort + self.temperature_bandwidth:
            action = self.previous_action.copy()
        else: 
            action[self.controlled_heater_name] = 0
        if external_input > 0:
            action[self.controlled_heater_name] = max(external_input, action[self.controlled_heater_name])
        action = self.check_time_elapsed_since_last_state_change(state, action)
        self.previous_action = action
        return action


class ChargeController(Controller):
    battery_name: str
    battery_charger_name: str
    SOC_max: float
    SOC_min: float
    baseline_battery_efficiency: float
    baseline_inverter_efficiency: float
    def __init__(self, name, 
                 battery_name: str,   
                 battery_SOC_sensor_name: str, 
                 AC_output_sensor_name: str | None = None,
                 PV_power_sensor_name: str | None = None,
                 SOC_min: float = 0.3, 
                 SOC_max: float = 0.9, 
                 baseline_battery_efficiency: float = 0.9,
                 baseline_inverter_efficiency: float = 0.92):
        self.battery_name = battery_name
        self.battery_charger_name = f'{battery_name}_charger'
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.baseline_battery_efficiency = baseline_battery_efficiency
        self.baseline_inverter_efficiency = baseline_inverter_efficiency
        sensors = {'battery SOC': battery_SOC_sensor_name}
        if PV_power_sensor_name is not None:
            sensors['PV power'] = PV_power_sensor_name
        if AC_output_sensor_name is not None:
            sensors['output power'] = AC_output_sensor_name
        super().__init__(name, controlled_components=[self.battery_charger_name, self.battery_name], sensors = sensors)
    
    def initialize(self, ctx):
        super().initialize(ctx)      

    def _compute_action(self, state: SimulationState):
        # In the case of the inverter, the action is the energy required to balance the controlled sensor node (normally the exchange with the grid)
        # This involves two checks:
        #   - Power check (the power should not be higher than what is allowed by the battery)
        #   - Energy check (we should not be asking from the battery more energy then what is stored inside)
        PV_power_input = self.obs['PV power'] if 'PV power' in self.obs.keys() else 0.0
        AC_output = self.obs['output power'] if 'output power' in self.obs.keys() else 0.0
        DC_output = AC_output / self.baseline_inverter_efficiency
        DC_balance = PV_power_input + DC_output
        SOC = self.obs['battery SOC']
        # Deriving action. NOTE: Positive action means charging the battery
        if DC_balance >= 0:  # If the node balance is positive, the inverter will try to charge the battery
            energy_to_charge = min(self.controlled_components[self.battery_name].charger.get_maximum_charge_power(), DC_balance) * state.time_step  # First we limit based on battery power limits
            action_value = min(self.controlled_components[self.battery_name].battery_pack.max_capacity * (self.SOC_max - SOC) * self.baseline_battery_efficiency, energy_to_charge) / state.time_step # Then we limit based on the available energy
        else:
            energy_to_discharge = min(self.controlled_components[self.battery_name].charger.get_maximum_discharge_power(), -DC_balance) * state.time_step # First we limit based on battery power limits
            action_value = -min(self.controlled_components[self.battery_name].battery_pack.max_capacity * (SOC - self.SOC_min) * self.baseline_battery_efficiency, energy_to_discharge) / state.time_step # Then we limit based on the available energy
        action = {self.battery_charger_name: action_value}
        self.previous_action = action
        return action