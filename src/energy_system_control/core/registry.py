from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SignalKey:
    main_key: str
    secondary_key: str      # e.g. "electric", "heat"

class SignalRegistry:
    def __init__(self):
        self._key_to_col: dict[SignalKey, int] = {}
        self._col_to_key: list[SignalKey] = []

    def register(self, main_key: str, secondary_key: str) -> int:
        key = SignalKey(main_key, secondary_key)
        col = len(self._col_to_key)
        self._key_to_col[key] = col
        self._col_to_key.append(key)
        return col

    def col_index(self, main_key: str, secondary_key: str) -> int:
        return self._key_to_col[SignalKey(main_key, secondary_key)]

class SimulationData:
    ports: np.array
    sensors: np.array
    controllers: np.array

    def __init__(self):
        self.ports = None
        self.sensors = None
        self.controllers = None

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