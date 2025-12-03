from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SimulationData:
    ports: np.array = None
    sensors: np.array = None
    controllers: np.array = None

    def create_empty_datasets(self, time_vector, signal_registry_ports, signal_registry_controllers, signal_registry_sensors):
        self.ports = np.empty((len(time_vector), len(signal_registry_ports._col_to_key)), dtype=np.float32)
        self.controllers = np.empty((len(time_vector), len(signal_registry_controllers._col_to_key)), dtype=np.float32)
        self.sensors = np.empty((len(time_vector), len(signal_registry_sensors._col_to_key)), dtype=np.float32)

    def to_dataframe(self, time_vector, signal_registry_ports, signal_registry_controllers, signal_registry_sensors):
        columns = [f'{key.main_key}:{key.secondary_key}' for key in signal_registry_ports._col_to_key]
        df_ports = pd.DataFrame(self.ports, columns = columns, index=pd.Index(time_vector / 3600, name='time'))
        columns = [f'{key.main_key}:{key.secondary_key}' for key in signal_registry_controllers._col_to_key]
        df_controllers = pd.DataFrame(self.controllers, columns = columns, index=pd.Index(time_vector / 3600, name='time'))
        columns = [f'{key.main_key}' for key in signal_registry_sensors._col_to_key]
        df_sensors = pd.DataFrame(self.sensors, columns = columns, index=pd.Index(time_vector / 3600, name='time'))
        return df_ports, df_controllers, df_sensors