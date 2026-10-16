import time
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

from energy_system_control.core.base_environment import RealTimeControlEnvironment
from energy_system_control.sim.config import RealTimeControlConfig
from energy_system_control.sim.simulation_data import SimulationData  # wherever it lives
from energy_system_control.sim.state import RealTimeControlState
from energy_system_control.sim.results import SimulationResults
from energy_system_control.core.base_classes import Manager


@dataclass
class RealTimeControl(Manager):
    env: RealTimeControlEnvironment
    cfg: RealTimeControlConfig


    def run(self, verbose: bool = False) -> SimulationData:
        self.verbose = verbose
        if self.verbose is True:
            print(f'Starting control loop')
        self.state = RealTimeControlState(
            start_datetime = self.cfg.simulation_start_datetime.replace(second=0, microsecond=0)
        )
        self.state.initialize(self.cfg)
        self.env.initialize(self.state, self.cfg)  # This allows the environment to initialize the provider if needed

        # Prepare simulation data storage
        sim_data = SimulationData()
        sim_data.create_empty_datasets(
            self.state.scheduled_time_vector,
            self.env.signal_registry_ports,
            self.env.signal_registry_controllers,
            self.env.signal_registry_sensors,
        )
        # Read any time series data once
        # self._read_timeseries_data()
        # Initialize units / reset components, controllers, sensors
        self._initialize_units()

        # Align the first execution to the next wall-clock boundary.
        next_run = self.state.current_datetime
        current_time = datetime.now().astimezone()
        start_time_mono = time.monotonic()

        # Convert the initial wall-clock delay to a monotonic deadline.
        delay = (next_run - current_time).total_seconds()
        deadline = time.monotonic() + max(0.0, delay)

        while current_time < self.cfg.simulation_end_datetime:
            # Wait until the scheduled instant.
            time.sleep(max(0.0, deadline - time.monotonic()))

            # Now calculate actual execution time and save it
            self.state.actual_time = datetime.now().astimezone()
            self.state.actual_time_vector[self.state.time_id] = self.state.actual_time
            if self.verbose is True:
                print(f'Running step at scheduled time {deadline} and actual time {self.state.actual_time}')
            # Run the control cycle
            try:
                self.run_control_cycle(sim_data)
            except Exception:
                pass 

            deadline += self.cfg.time_step_s

            # If execution overran, skip missed slots rather than
            # executing several control cycles back-to-back.
            now_mono = time.monotonic()
            missed = 0
            if deadline <= now_mono:
                missed = int((now_mono - deadline) // self.cfg.time_step_s) + 1
                deadline += missed * self.cfg.time_step_s
                # next_run += timedelta(seconds=missed * self.cfg.time_step_s)

            # Updating scheduled times
            self.state.time_id += missed + 1
            self.state.time += self.cfg.time_step_s * (missed + 1)

            
            # Saving the new "now" time
            current_time = datetime.now().astimezone()

        simulation_results = SimulationResults(sim_data, 
                                                self.state.time_step, 
                                                self.state.actual_time_vector,
                                                self.env.signal_registry_ports,
                                                self.env.signal_registry_controllers,
                                                self.env.signal_registry_sensors,)
        return simulation_results

    def run_control_cycle(self, sim_data: SimulationData):

        # 1 - Collect measurements from all sensors
        self._collect_measurements()

        # 2. Initialize control actions
        self.state.control_actions = {}

        # 3. Controllers compute actions
        self._get_controller_actions()
        if self.verbose:
            for controlled_component, action in self.state.control_actions.items():
                print(f'Controller {controlled_component} is sent action {action}')


        # 4. Actuators send control actions to the system
        self._operate_actuators()

        # 4. Save results for this step
        sim_data = self._save_simulation_data(sim_data)


    def _operate_actuators(self):
        for actuator_name, actuator in self.env.actuators.items():
            action = self.state.control_actions.get(actuator.name)
            actuator.send_control_signal(self.state, action)
