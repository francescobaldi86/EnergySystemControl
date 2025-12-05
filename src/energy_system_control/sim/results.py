from energy_system_control.sim.simulation_data import SimulationData
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class SimulationResults:
    data: SimulationData
    time_step: float
    time_vector: np.array
    signal_registry_ports: Any
    signal_registry_controllers: Any
    signal_registry_sensors: Any

    def to_dataframe(self):
        return self.data.to_dataframe(self.time_vector, self.signal_registry_ports, self.signal_registry_controllers, self.signal_registry_sensors)

    def _get_cumulated_result(self, port_name: str, layer_name: str, scaling_factor: float = 1):
        col = self.signal_registry_ports.col_index(port_name, layer_name)
        return self.data.ports[:, col].sum() * self.time_step * scaling_factor

    def _get_cumulated_result_with_sign(self, port_name: str, layer_name: str, sign: str, scaling_factor: float = 1):
        col = self.signal_registry_ports.col_index(port_name, layer_name)
        match sign:
            case 'only positive':
                return self.data.ports[self.data.ports[:, col] >= 0.0, col].sum() * self.time_step * scaling_factor
            case 'only negative':
                return -self.data.ports[self.data.ports[:, col] <= 0.0, col].sum() * self.time_step * scaling_factor

    def get_cumulated_electricity(self, port_name: str, unit: str = "kWh", sign: str = "net"):
        match unit:
            case "kWh":
                scaling_factor = 1 / 3_600
            case "MWh":
                scaling_factor = 1 / 3_600_000
            case _:
                raise ValueError(unit)

        match sign:
            case "net":
                return self._get_cumulated_result(port_name, "electricity", scaling_factor)
            case "only positive" | "only negative":
                return self._get_cumulated_result_with_sign(port_name, "electricity", sign, scaling_factor)
