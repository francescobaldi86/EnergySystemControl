# energy_system_control/sim/simulator.py
from dataclasses import dataclass
from typing import Any
from numbers import Number
import numpy as np
import pandas as pd

from energy_system_control.core.base_environment import Environment  # or move Environment to core/model.py
from energy_system_control.core.base_classes import InitContext, EnvironmentalData
from .config import SimulationConfig
from .state import SimulationState
from energy_system_control.helpers import C2K
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
        self.env.initialize(self.state, self.cfg)  # This allows the environment to initialize the provider if needed
        # Compile the step plan once: all per-step lookups (port->column maps,
        # propagation pairs, balance checks, solar angles) are precomputed here.
        self.plan = self.env.build_step_plan(self.state, self.cfg)

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
        while self.state.time < (self.cfg.simulation_end_h * 3600.0 - 1e-9):
            self._step(sim_data)
            self.state.time += self.cfg.time_step_s
            self.state.time_id += 1

        # Creating results object
        if self.cfg.simulation_start_datetime is not None:
            results_index = pd.date_range(start = self.cfg.simulation_start_datetime,
                                          end = self.cfg.simulation_start_datetime + pd.Timedelta(hours = self.cfg.simulation_end_h),
                                          freq = pd.Timedelta(hours = self.cfg.time_step_h),
                                          inclusive = "left"
                                          )
        else:
            results_index = self.state.time_vector

        simulation_results = SimulationResults(sim_data, 
                                               self.state.time_step, 
                                               results_index,
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
                component.resample_data(
                    time_step_h = self.cfg.time_step_h, 
                    simulation_end_h = self.cfg.simulation_end_h + self.cfg.prediction_horizon_margin_h,
                    simulation_start_datetime = self.cfg.simulation_start_datetime)
        for _, predictor in self.env.predictors.items():
            if callable(getattr(predictor, 'resample_data', None)):
                predictor.resample_data(
                    time_step_h = self.cfg.time_step_h, 
                    simulation_end_h = self.cfg.simulation_end_h + self.cfg.prediction_horizon_margin_h,
                    simulation_start_datetime = self.cfg.simulation_start_datetime)

    def _normalize_measurement(self, value):
        """
        Normalize a measurement value to a single scalar.
        - Numbers (int, float, np scalar types) → saved directly
        - Iterables (list, np.array, etc.) → average is saved
        - None → NaN is saved
        """
        # Handle None explicitly
        if value is None:
            return np.nan
        
        # Handle numeric types (built-in + numpy)
        if isinstance(value, (Number, np.generic)):
            return float(value)
        
        # Handle iterables (lists, arrays, etc.)
        try:
            # Try to convert to array and compute mean
            arr = np.asarray(value)
            if arr.size == 0:
                return np.nan  # Empty array
            return float(np.mean(arr))
        except (TypeError, ValueError):
            # If it's not iterable or can't be converted, return NaN
            return np.nan

    def _step(self, sim_data: SimulationData) -> None:
        plan = self.plan
        state = self.state

        # 1. Update environmental data (this is time-varying state)
        state.environmental_data = self._update_environmental_data()

        # 2. Measure all sensors at time t
        # Measured at the beginning of the step so that controllers have access to the most
        # recent measurements when they calculate their actions, and so that sensor data
        # also exists for the initial state of the simulation.
        for sensor in plan.sensors:
            sensor.measure(environment=self.env, state=state)

        # 3. Reset port data (unconnected ports carry zero flow)
        for port in plan.ports:
            port.reset_flow_data()
        # Unconnected ports' flows are set to 0
        for port in plan.unconnected_ports:
            for layer in port.layers:
                port.flows[layer] = 0.0

        # 4. Initialize control actions
        state.control_actions = {}

        # 5. Assign fluid port values
        self._propagate_port_values()

        # 6. Controllers compute actions
        self._get_controller_actions()

        # 7. Simulate components in your chosen order
        self._simulate_all_components()

        # 8. Check balances on all nodes:
        self._check_connection_balance()  # This will raise an error if the balance is not correct

        # Save results for this step
        self._save_simulation_data(sim_data)

    def _update_environmental_data(self):
        env_data = self.state.environmental_data
        plan = self.plan

        # Overwrite with forecast / time series if available
        if self.env.environmental_data_provider:
            time_id = self.state.time_id
            env_data = self.env.environmental_data_provider.get_environmental_data(time_id, plan.datetimes[time_id])
            # Check for None values and substitute them with defaults
            defaults = self.cfg.environmental_defaults
            for field_name in plan.env_field_names:
                if getattr(env_data, field_name) is None:
                    setattr(env_data, field_name, getattr(defaults, field_name))

        # Solar angles were precomputed for the whole horizon in the step plan
        if plan.solar_zenith is not None:
            time_id = self.state.time_id
            if env_data.solar_zenith is None:
                env_data.solar_zenith = plan.solar_zenith[time_id]
            if env_data.solar_azimuth is None:
                env_data.solar_azimuth = plan.solar_azimuth[time_id]

        return env_data

    def _propagate_port_values(self):
        env_ports = self.env.ports
        state = self.state
        for component in self.plan.inherit:
            inherited_fluid_ports_info = component.set_inherited_fluid_port_values(state)
            for port_name, temperature in inherited_fluid_ports_info.items():
                if port_name and env_ports[port_name].connected_port is not None:
                    env_ports[port_name].connected_port.T = temperature
            inherited_heat_ports_info = component.set_inherited_heat_port_values(state)
            for port_name, temperature in inherited_heat_ports_info.items():
                if port_name and env_ports[port_name].connected_port is not None:
                    env_ports[port_name].connected_port.T = temperature

    def _simulate_all_components(self):
        plan = self.plan
        actions = self.state.control_actions
        for component in plan.explicit:
            self._take_component_step(component, actions.get(component.name))
        for component in plan.controlled:
            self._take_component_step(component, actions.get(component.name))
        self._solve_algebric_networks()
        for component in plan.storage:
            self._take_component_step(component, actions.get(component.name))
        for component in plan.grid:
            self._take_component_step(component, actions.get(component.name))

    def _solve_algebric_networks(self):
        # Iterate over a copy of the remaining components (no mutation during
        # iteration); each updated port is propagated immediately.
        remaining = list(self.plan.algebraic)
        ports = self.env.ports
        while remaining:
            number_of_updated_ports = 0
            next_remaining = []
            for component in remaining:
                solved, updated_ports = component.balance(self.state)
                for port in updated_ports:
                    ports[port].propagate_port_values()
                    number_of_updated_ports += 1
                if solved is not True:
                    next_remaining.append(component)
            if number_of_updated_ports == 0:
                raise RuntimeError(f"Could not solve the network at time {self.state.time}. Remaining components: {[comp.name for comp in next_remaining]}")
            remaining = next_remaining

    def _get_controller_actions(self):
        actions = {}
        for controller in self.plan.controllers:
            controller.get_obs(self.env, self.state)
            ctrl_actions = controller.get_action(self.state)
            for comp, action in ctrl_actions.items():
                if comp in actions:
                    raise ValueError(f"Component {comp} controlled by multiple controllers.")
                actions[comp] = action
        self.state.control_actions = actions

    def _take_component_step(self, component, action):
        component.step(self.state, action)
        # Update values of connected ports (precompiled port pairs from the step plan)
        for port, connected_port, layers, copy_T in self.plan.step_propagation[component.name]:
            flows = port.flows
            connected_flows = connected_port.flows
            for layer in layers:
                connected_flows[layer] = -flows[layer]
            if copy_T:
                connected_port.T = port.T

    def _check_connection_balance(self):
        # Checks that all connections have the same flow on both sides
        for port_a, port_b, layer in self.plan.balance_checks:
            value_a = port_a.flows[layer]
            value_b = port_b.flows[layer]
            if abs(value_a + value_b) > 1e-5:
                raise ValueError(f"Connection ({port_a.name}, {port_b.name}) has unbalanced flows: {value_a:.2f} != {value_b:.2f}")

    def _save_simulation_data(self, sim_data):
        time_id = self.state.time_id
        # Ports: one row write; the plan is aligned with the array columns by
        # construction. 'temperature' is the extra FluidPort registration and
        # reads port.T instead of port.flows.
        ports_row = []
        for port, layer in self.plan.port_save:
            value = port.T if layer == 'temperature' else port.flows[layer]
            ports_row.append(np.nan if value is None else value)
        sim_data.ports[time_id, :] = ports_row
        # Controllers
        controllers_arr = sim_data.controllers
        for controller, col_map, reward_col, td_error_col in self.plan.controller_save:
            for component_name, action_value in controller.previous_action.items():
                controllers_arr[time_id, col_map[component_name]] = action_value
            if reward_col is not None:
                controllers_arr[time_id, reward_col] = controller.agent.last_reward
                controllers_arr[time_id, td_error_col] = controller.agent.last_td_error
        # Sensors
        sim_data.sensors[time_id, :] = [self._normalize_measurement(sensor.current_measurement)
                                        for sensor in self.plan.sensor_save]
        return sim_data