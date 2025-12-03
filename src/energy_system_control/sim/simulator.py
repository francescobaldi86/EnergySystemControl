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
        self._initialize_units()

        # Read any time series data once
        self._read_timeseries_data()

        # Main loop
        while state.time < (self.cfg.time_end_h * 3600.0 - 1e-9):
            self._step(state, sim_data)
            state.time += self.cfg.time_step_s
            state.time_id += 1

        return sim_data

    def _step(self, state: SimulationState, sim_data: SimulationData) -> None:
        env = self.env  # just a shorthand

        # Reset port flows and sensor measurements
        for _, port in env.ports.items():
            port.reset_flow_data()
        for _, sensor in env.sensors.items():
            sensor.current_measurement = None

        # Initialize control actions and list of components to simulate
        state.control_actions = {}
        components_to_simulate = list(env.components.keys())

        # Update environmental data (this is time-varying state)
        state.environmental_data = self._update_environmental_data(state)

        # Let components see environment data, if needed
        self._update_environmental_data()

        # Assign fluid port values
        self._propagate_fluid_port_values()

        # Simulate components in your chosen order
        self._simulate_components_of_type("Demand", components_to_simulate, state)
        self._simulate_components_of_type("Producer", components_to_simulate, state)

        # Controllers + controlled components
        self._get_controller_actions(components_to_simulate, state)

        self._simulate_components_of_type("Utility", components_to_simulate, state)
        self._simulate_components_of_type("StorageUnit", components_to_simulate, state)
        self._simulate_components_of_type("BalancingUtility", components_to_simulate, state)

        if components_to_simulate:
            raise RuntimeError(
                f"Step concluded but components {components_to_simulate} were not simulated "
                f"at time {state.time}, time ID {state.time_id}."
            )

        # Save results for this step
        self._save_simulation_data(state, sim_data)

    def _update_environmental_data(self, state: SimulationState):
        return {
            "Temperature cold water": C2K(15),
            "Temperature ambient": C2K(20),
        }

    def _propagate_fluid_port_values(self):
        from energy_system_control.core.ports import FluidPort
        env = self.env
        for _, component in env.components.items():
            port_name, T = component.set_inherited_fluid_port_values()
            if port_name:
                env.ports[env.ports[port_name].connected_port].T = T

    def _simulate_components_of_type(self, type: str):
        components = self.components_classified[type]
        for component in components:
            if component.name in self.components_to_simulate:
                component.time = self.time
                component.time_id = self.time_id
                self._take_component_step(component, None)           
                self.components_to_simulate.remove(component.name)

    def _get_controller_actions(self):
        for controller_name in self.ordered_controllers:
            self.controllers[controller_name].time = self.time
            self.controllers[controller_name].time_id = self.time_id
            self.controllers[controller_name].get_obs(self)
            actions = self.controllers[controller_name].get_action()
            # After a controller has been simulated, its controlled component is immediately simulted as well
            for component_name, action in actions.items():
                if component_name in self.components_to_simulate:
                    self._take_component_step(self.components[component_name], action)
                    self.components_to_simulate.remove(component_name)
                else:
                    raise(KeyError, f'Component {component_name} has been simulated before its related control action was calculated at time {self.time}, time ID {self.time_id}. Check what happens!')
            self.control_actions.update(actions)

    def _take_component_step(self, component, action):
        component.step(action)
        # Update values of connected ports
        for _, port in component.ports.items():
            for layer, value in port.flow.items():
                if port.connected_port:
                    self.ports[port.connected_port].flow[layer] = -value
                    if isinstance(self.ports[port.connected_port], FluidPort):
                        self.ports[port.connected_port].T = self.ports[port.name].T

