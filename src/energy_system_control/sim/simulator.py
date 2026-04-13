# energy_system_control/sim/simulator.py
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from energy_system_control.core.base_environment import Environment  # or move Environment to core/model.py
from energy_system_control.core.base_classes import InitContext, EnvironmentalData
from .config import SimulationConfig
from .state import SimulationState
from energy_system_control.helpers import C2K, calculate_solar_angles
from energy_system_control.core.port import FluidPort, HeatPort
from energy_system_control.sim.simulation_data import SimulationData  # wherever it lives
from energy_system_control.sim.results import SimulationResults
from energy_system_control.controllers.RL.RLcontrollers import RLController

@dataclass
class Simulator:
    env: Environment
    cfg: SimulationConfig

    def run(self) -> SimulationData:
        self.state = SimulationState()
        self.state.initialize(self.cfg)
        self.env.initialize(self.state)  # This allows the environment to initialize the provider if needed

        # Prepare simulation data storage
        sim_data = SimulationData()
        sim_data.create_empty_datasets(
            self.state.time_vector,
            self.env.signal_registry_ports,
            self.env.signal_registry_controllers,
            self.env.signal_registry_sensors,
        )
        # Read any time series data once
        self._read_timeseries_data()
        # Initialize units / reset components, controllers, sensors
        self._initialize_units()        

        # Main loop
        while self.state.time < (self.cfg.time_end_h * 3600.0 - 1e-9):
            self._step(sim_data)
            self.state.time += self.cfg.time_step_s
            self.state.time_id += 1

        # Creating results object
        simulation_results = SimulationResults(sim_data, 
                                               self.state.time_step, 
                                               self.state.time_vector,
                                               self.env.signal_registry_ports,
                                               self.env.signal_registry_controllers,
                                               self.env.signal_registry_sensors,)
        return simulation_results
    
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
        

    def _read_timeseries_data(self):
        # Read timeseries data from components
        for _, component in self.env.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.cfg.time_step_h, self.cfg.time_end_h + self.cfg.prediction_horizon_margin_h)

    def _step(self, sim_data: SimulationData) -> None:
        env = self.env  # just a shorthand
        
        # 1. Update environmental data (this is time-varying state)
        self.state.environmental_data = self._update_environmental_data()

        # 2. Measure all sensors at time t
        for _, sensor in env.sensors.items():
            sensor.measure(environment=env, state=self.state)  # We measure all sensors at the beginning of the step to make sure that controllers have access to the most recent measurements when they calculate their actions. This also ensures that we have sensor data for the initial state of the simulation.
        
        # 3. Reset port data
        for _, port in env.ports.items():
            port.reset_flow_data()
            port.reset_state_value()
        
        # 4. Initialize control actions
        self.state.control_actions = {}

        # 5. Assign fluid port values
        self._propagate_port_values()

        # 6. Controllers compute actions
        self._get_controller_actions()

        # 7. Simulate components in your chosen order
        self._simulate_all_components()

        # 8. Check balances on all nodes:
        self._check_connection_balance()  # This will raise an error if the balance is not correc

        if self.components_to_simulate:
            raise RuntimeError(
                f"Step concluded but components {self.components_to_simulate} were not simulated "
                f"at time {self.state.time}, time ID {self.state.time_id}."
            )

        # Save results for this step
        sim_data = self._save_simulation_data(sim_data)

    def _update_environmental_data(self):
        env_data = self.state.environmental_data

        # Example: overwrite with forecast / time series if available
        if self.env.environmental_data_provider:
            env_data = self.env.environmental_data_provider.get_environmental_data(self.state.time_id, self.state.simulation_start_datetime + pd.to_timedelta(self.state.time, unit='s'))

        # Automatically compute solar angles if not available
        if env_data.solar_zenith is None or env_data.solar_azimuth is None:
            if self.env.latitude and self.env.longitude:
                dt = self.state.simulation_start_datetime + pd.to_timedelta(self.state.time, unit='s')
                env_data.solar_zenith, env_data.solar_azimuth = calculate_solar_angles(self.env.latitude, self.env.longitude, dt)

        return env_data

    def _propagate_port_values(self):
        env = self.env
        for _, component in env.components.items():
            port_name, T = component.set_inherited_fluid_port_values(self.state)
            if port_name and env.ports[port_name].connected_port is not None:
                env.ports[env.ports[port_name].connected_port].T = T
            port_name, T = component.set_inherited_heat_port_values(self.state)
            if port_name and env.ports[port_name].connected_port is not None:
                env.ports[env.ports[port_name].connected_port].T = T
    
    def _simulate_all_components(self):
        self.components_to_simulate = list(self.env.components.keys())
        self._simulate_components_of_type("ExplicitComponent")
        self._simulate_components_of_type("ControlledComponent")
        self._solve_algebric_networks()
        self._simulate_components_of_type("Grid")
        self._simulate_components_of_type("StorageUnit")

    def _simulate_components_of_type(self, type: str):
        components = self.env.components_classified[type]
        for component in components:
            if component.name in self.components_to_simulate:
                action = self.state.control_actions.get(component.name)
                self._take_component_step(component, action)           
                self.components_to_simulate.remove(component.name)

    def _solve_algebric_networks(self):
        components_to_simulate = self.env.components_classified['Bus'] + self.env.components_classified['ImplicitComponent']
        while len(components_to_simulate) > 0:
            updated_ports = 0
            for component in components_to_simulate:
                solved, updated_ports = component.balance(self.state)
                if solved is True:
                    components_to_simulate.remove(component)
                for port in updated_ports:
                    port.propagate_port_values()
                    updated_ports += 1
            if updated_ports == 0:
                raise RuntimeError(f"Could not solve the network at time {self.state.time}. Remaining components: {[comp.name for comp in components_to_simulate]}")
                

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

    def _take_component_step(self, component, action):
        component.step(self.state, action)
        # Update values of connected ports
        for _, port in component.ports.items():
            for layer, value in port.flow.items():
                if port.connected_port:
                    self.env.ports[port.connected_port].flow[layer] = -value
                    if isinstance(self.env.ports[port.connected_port], FluidPort | HeatPort):
                        self.env.ports[port.connected_port].T = self.env.ports[port.name].T

    def _check_connection_balance(self):
        # Checks that all connections have the same flow on both sides
        env = self.env
        for connection in env.connections:
            if env.ports[connection[0]].flows != env.ports[connection[1]].flows:
                raise ValueError(f"Connection {connection} has unbalanced flows: {env.ports[connection[0]].flows} != {env.ports[connection[1]].flows}")


    def _save_simulation_data(self, sim_data):
        # Ports
        time_id = self.state.time_id
        for port_name, port in self.env.ports.items():
            for layer, flow in port.flow.items():
                col = self.env.signal_registry_ports.col_index(port_name, layer)
                sim_data.ports[time_id, col] = flow / self.state.time_step
            if isinstance(port, FluidPort):
                col = self.env.signal_registry_ports.col_index(port_name, 'temperature')
                sim_data.ports[time_id, col] = port.T
        # Controllers
        for controller_name, controller in self.env.controllers.items():
            for component_name in controller.controlled_component_names:
                col = self.env.signal_registry_controllers.col_index(controller_name, component_name)
                sim_data.controllers[time_id, col] = controller.previous_action[component_name]
                if isinstance(controller, RLController):
                    col_reward = self.env.signal_registry_controllers.col_index(controller_name, 'reward')
                    col_td_error = self.env.signal_registry_controllers.col_index(controller_name, 'td_error')
                    sim_data.controllers[time_id, col_reward] = controller.agent.last_reward
                    sim_data.controllers[time_id, col_td_error] = controller.agent.last_td_error
        # We also create a registry for each sensor
        for sensor_name, sensor in self.env.sensors.items():
            col = self.env.signal_registry_sensors.col_index(sensor_name, "")
            sim_data.sensors[time_id, col] = sensor.current_measurement
        return sim_data