from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, fields
from numpy import nan

@dataclass(frozen=True, slots=True)
class InitContext:
    environment: Any
    state: Any
    rng: Optional[Any] = None
    logger: Optional[Any] = None
    config: Optional[dict] = None


@dataclass
class EnvironmentalData:
    temperature_ambient: float = 293.15  # K
    temperature_cold_water: float = 288.15  # K
    direct_irradiation: float = 0.0      # W/m^2
    diffuse_irradiation: float = 0.0     # W/m^2
    solar_zenith: float | None = None           # degrees
    solar_azimuth: float | None = None         # degrees       

@dataclass
class Manager:
    env: Any

    def _initialize_units(self):
            ctx = InitContext(environment=self.env, state=self.state)
            for _, component in self.env.components.items():
                component.initialize(ctx)
            for _, port in self.env.ports.items():
                port.initialize(ctx)
            for _, sensor in self.env.sensors.items():
                sensor.initialize(ctx)
            for _, predictor in self.env.predictors.items():
                predictor.initialize(ctx)
            for _, controller in self.env.controllers.items():
                controller.initialize(ctx)
            for _, actuator in self.env.actuators.items():
                actuator.initialize(ctx)

    def _collect_measurements(self):
        for _, sensor in self.env.sensors.items():
            sensor.measure(environment=self.env, state=self.state)  # We measure all sensors at the beginning of the step to make sure that controllers have access to the most recent measurements when they calculate their actions. This also ensures that we have sensor data for the initial state of the simulation.

    def _get_controller_actions(self):
        actions = {}
        for controller_name in self.env.ordered_controllers:
            controller = self.env.controllers[controller_name]
            controller.get_obs(self.env, self.state)
            ctrl_actions = controller.get_action(self.state)
            for comp, action in ctrl_actions.items():
                if comp in actions:
                    raise ValueError(f"Component {comp} controlled by multiple controllers.")
                actions[comp] = action
        self.state.control_actions = actions

    def _save_simulation_data(self, sim_data):
        # Ports
        time_id = self.state.time_id
        for port_name, port in self.env.ports.items():
            for layer, flow in port.flows.items():
                col = self.env.signal_registry_ports.col_index(port_name, layer)
                try:
                    sim_data.ports[time_id, col] = flow if flow is not None else nan
                except ValueError as e:
                    print(f"Error saving flow for port {port_name} and layer {layer}: {e}")
            if getattr(port, "T") is not None:
                col = self.env.signal_registry_ports.col_index(port_name, 'temperature')
                sim_data.ports[time_id, col] = port.T
        # Controllers
        for controller_name, controller in self.env.controllers.items():
            for controlled_component_name, action_value in controller.previous_action.items():
                col = self.env.signal_registry_controllers.col_index(controller_name, controlled_component_name)
                sim_data.controllers[time_id, col] = action_value
                # if isinstance(controller, RLController):
                #    col_reward = self.env.signal_registry_controllers.col_index(controller_name, 'reward')
                #    col_td_error = self.env.signal_registry_controllers.col_index(controller_name, 'td_error')
                #    sim_data.controllers[time_id, col_reward] = controller.agent.last_reward
                #    sim_data.controllers[time_id, col_td_error] = controller.agent.last_td_error
        # We also create a registry for each sensor
        for sensor_name, sensor in self.env.sensors.items():
            col = self.env.signal_registry_sensors.col_index(sensor_name, "")
            sim_data.sensors[time_id, col] = self._normalize_measurement(sensor.current_measurement)
        return sim_data