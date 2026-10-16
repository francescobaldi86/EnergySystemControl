from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SimulationData:
    ports: np.array = None
    sensors: np.array = None
    controllers: np.array = None
    rl: np.array = None

    def create_empty_datasets(self, time_vector, signal_registry_ports, signal_registry_controllers, signal_registry_sensors, signal_registry_rl = None):
        self.ports = np.empty((len(time_vector), len(signal_registry_ports._col_to_key)), dtype=np.float32)
        self.controllers = np.empty((len(time_vector), len(signal_registry_controllers._col_to_key)), dtype=np.float32)
        self.sensors = np.empty((len(time_vector), len(signal_registry_sensors._col_to_key)), dtype=np.float32)
        if signal_registry_rl:
            self.rl = np.empty((len(time_vector), len(signal_registry_rl._col_to_key)), dtype=np.float32)

    def to_dataframe(self, time_vector, signal_registry_ports, signal_registry_controllers, signal_registry_sensors, scheduled_time_vector = None):
        index = pd.Index(pd.DatetimeIndex(time_vector).strftime("%Y-%m-%d %H:%M:%S"), name = 'time')
        columns = [f'{key.main_key}:{key.secondary_key}' for key in signal_registry_ports._col_to_key]
        df_ports = pd.DataFrame(self.ports, columns = columns, index=index)
        columns = [f'{key.main_key}:{key.secondary_key}' for key in signal_registry_controllers._col_to_key]
        df_controllers = pd.DataFrame(self.controllers, columns = columns, index=index)
        columns = [f'{key.main_key}' for key in signal_registry_sensors._col_to_key]
        df_sensors = pd.DataFrame(self.sensors, columns = columns, index=index)
        if scheduled_time_vector is not None:
            try:
                df_ports.insert(0, 'scheduled time', scheduled_time_vector)
                df_controllers.insert(0, 'scheduled time', scheduled_time_vector)
                df_sensors.insert(0, 'scheduled time', scheduled_time_vector)
            except ValueError as e:
                print(f'Could not include scheduled time vector, error: {e}')
        return df_ports, df_controllers, df_sensors