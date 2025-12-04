# energy_system_control/sim/simulator.py
from dataclasses import dataclass
from typing import Any
import numpy as np

from energy_system_control.core.base_environment import Environment  # or move Environment to core/model.py
from .config import SimulationConfig
from .state import SimulationState
from energy_system_control.helpers import C2K
from energy_system_control.core.ports import FluidPort
from energy_system_control.sim.simulation_data import SimulationData  # wherever it lives

@dataclass
class Simulator:
    env: Environment
    cfg: SimulationConfig

    def run(self) -> SimulationData:
        state = SimulationState()
        state.init_time_vector(self.cfg)

        # Prepare simulation data storage
        sim_data = SimulationData()
        sim_data.create_empty_datasets(
            state.time_vector,
            self.env.signal_registry_ports,
            self.env.signal_registry_controllers,
            self.env.signal_registry_sensors,
        )

        # Initialize units / reset components, controllers, sensors
        self._initialize_units(state)

        # Read any time series data once
        self._read_timeseries_data()

        # Main loop
        while state.time < (self.cfg.time_end_h * 3600.0 - 1e-9):
            self._step(state, sim_data)
            state.time += self.cfg.time_step_s
            state.time_id += 1

        return sim_data
    
    def _initialize_units(self, state:SimulationState):
        env = self.env
        time_step = self.cfg.time_step_s
        for _, component in env.components.items():
            component.time_step = time_step
            component.initialize(state)
        for _, controller in env.controllers.items():
            controller.time_step = time_step
            controller.initialize()
        for _, sensor in env.sensors.items():
            sensor.time_step = time_step
            sensor.initialize()

    def _read_timeseries_data(self):
        for _, component in self.env.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.cfg.time_step_s, self.cfg.time_end_s)

    def _step(self, state: SimulationState, sim_data: SimulationData) -> None:
        env = self.env  # just a shorthand

        # Reset port flows and sensor measurements
        for _, port in env.ports.items():
            port.reset_flow_data()
        for _, sensor in env.sensors.items():
            sensor.current_measurement = None

        # Initialize control actions and list of components to simulate
        state.control_actions = {}
        self.components_to_simulate = list(env.components.keys())

        # Update environmental data (this is time-varying state)
        state.environmental_data = self._update_environmental_data(state)

        # Let components see environment data, if needed
        self._update_environmental_data(state)

        # Assign fluid port values
        self._propagate_fluid_port_values(state)

        # Simulate components in your chosen order
        self._simulate_components_of_type("Demand", state)
        self._simulate_components_of_type("Producer", state)

        # Controllers + controlled components
        self._get_controller_actions(state)

        self._simulate_components_of_type("Utility", state)
        self._simulate_components_of_type("StorageUnit", state)
        self._simulate_components_of_type("BalancingUtility", state)

        if self.components_to_simulate:
            raise RuntimeError(
                f"Step concluded but components {self.components_to_simulate} were not simulated "
                f"at time {state.time}, time ID {state.time_id}."
            )

        # Save results for this step
        sim_data = self._save_simulation_data(state, sim_data)

    def _update_environmental_data(self, state: SimulationState):
        state.environmental_data = {
            "Temperature cold water": C2K(15),
            "Temperature ambient": C2K(20),
        }

    def _propagate_fluid_port_values(self, state: SimulationState):
        env = self.env
        for _, component in env.components.items():
            port_name, T = component.set_inherited_fluid_port_values(state)
            if port_name:
                env.ports[env.ports[port_name].connected_port].T = T

    def _simulate_components_of_type(self, type: str, state: SimulationState):
        components = self.env.components_classified[type]
        for component in components:
            if component.name in self.components_to_simulate:
                self._take_component_step(component, None, state)           
                self.components_to_simulate.remove(component.name)

    def _get_controller_actions(self, state: SimulationState):
        for controller_name in self.env.ordered_controllers:
            self.controllers[controller_name].get_obs(self, state)
            actions = self.controllers[controller_name].get_action(state)
            # After a controller has been simulated, its controlled component is immediately simulted as well
            for component_name, action in actions.items():
                if component_name in self.components_to_simulate:
                    self._take_component_step(self.components[component_name], action, state)
                    self.components_to_simulate.remove(component_name)
                else:
                    raise(KeyError, f'Component {component_name} has been simulated before its related control action was calculated at time {self.time}, time ID {self.time_id}. Check what happens!')
            # self.control_actions.update(actions)

    def _take_component_step(self, component, action, state):
        component.step(action, state)
        # Update values of connected ports
        for _, port in component.ports.items():
            for layer, value in port.flow.items():
                if port.connected_port:
                    self.ports[port.connected_port].flow[layer] = -value
                    if isinstance(self.ports[port.connected_port], FluidPort):
                        self.ports[port.connected_port].T = self.ports[port.name].T

    def _save_simulation_data(self, sim_data):
        # Ports
        for port_name, port in self.env.ports.items():
            for layer, flow in port.flow.items():
                col = self.env.signal_registry_ports.col_index(port_name, layer)
                sim_data.ports[self.time_id, col] = flow / self.time_step
            if isinstance(port, FluidPort):
                col = self.env.signal_registry_ports.col_index(port_name, 'temperature')
                sim_data.ports[self.time_id, col] = port.T
        # Controllers
        for controller_name, controller in self.env.controllers.items():
            for component_name in controller.controlled_component_names:
                col = self.env.signal_registry_controllers.col_index(controller_name, component_name)
                sim_data.controllers[self.time_id, col] = controller.previous_action[component_name]
        # We also create a registry for each sensor
        for sensor_name, sensor in self.env.sensors.items():
            col = self.env.signal_registry_sensors.col_index(sensor_name, "")
            if not sensor.current_measurement:
                sensor.get_measurement(self)  # This is needed in case no other component has used the sensor
            sim_data.sensors[self.time_id, col] = sensor.current_measurement
        return sim_data