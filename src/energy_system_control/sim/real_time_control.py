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


    def run(self) -> SimulationData:
        self.state = RealTimeControlState(
            simulation_start_datetime = self.cfg.simulation_start_datetime.replace(second=0, microsecond=0)
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
        now = datetime.now().astimezone()
        next_run = now.replace(second=0, microsecond=0)
        control_cycle_end = self.cfg.simulation_end_datetime

        step = timedelta(seconds=self.cfg.time_step_s)

        # Convert the initial wall-clock delay to a monotonic deadline.
        delay = (next_run - datetime.now().astimezone()).total_seconds()
        deadline = time.monotonic() + max(0.0, delay)

        while now < control_cycle_end:
            # Wait until the scheduled instant.
            now = datetime.now().astimezone()
            time.sleep(max(0.0, deadline - time.monotonic()))

            self.state.scheduled_time = next_run
            self.state.actual_start = datetime.now().astimezone()

            try:
                self.run_control_cycle()
            except Exception:
                # Replace with proper logging and fault handling.
                # log_error()
                pass 

            # Advance according to the schedule, not execution duration.
            next_run += step
            deadline += self.cfg.time_step_s

            # If execution overran, skip missed slots rather than
            # executing several control cycles back-to-back.
            now_mono = time.monotonic()
            if deadline <= now_mono:
                missed = int((now_mono - deadline) // self.cfg.time_step_s) + 1
                deadline += missed * self.cfg.time_step_s
                next_run += timedelta(seconds=missed * self.cfg.time_step_s)

        # Creating results object
        results_index = self.state.actual_time_vector

        simulation_results = SimulationResults(sim_data, 
                                                self.state.scheduled_time_step_s, 
                                                results_index,
                                                self.env.signal_registry_ports,
                                                self.env.signal_registry_controllers,
                                                self.env.signal_registry_sensors,)
        return simulation_results

    def run_control_cycle(self):

        # 1 - Collect measurements from all sensors
        self._collect_measurements()

        # 2. Initialize control actions
        self.state.control_actions = {}

        # 3. Controllers compute actions
        self._get_controller_actions()

        # 4. Actuators send control actions to the system
        self._operate_actuators()

        # 4. Save results for this step
        sim_data = self._save_simulation_data(sim_data)


    def _operate_actuators(self):
        for actuator_name, actuator in self.env.actuators.items():
            action = self.state.control_actions.get(actuator.name)
            actuator.send_control_signal(self.state, action)
