from energy_system_control.sim.simulation_data import SimulationData
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class SimulationResults:
    data: SimulationData
    time_step: float
    signal_registry_ports: Any
    signal_registry_controllers: Any
    signal_registry_sensors: Any

    def get_cumulated_result(self, port_name: str, layer_name: str, scaling_factor: float = 1):
        col = self.signal_registry_ports.col_index(port_name, layer_name)
        return self.data.ports[:, col].sum() * self.time_step * scaling_factor

    def get_cumulated_electricity(self, port_name: str, unit: str = "kWh", sign: str = "net"):
        match unit:
            case "kWh":
                scaling_factor = 1 / 3600
            case "MWh":
                scaling_factor = 1 / 3_600_000
            case _:
                raise ValueError(unit)

        match sign:
            case "net":
                return self.get_cumulated_result(port_name, "electricity", scaling_factor)
            case "only positive" | "only negative":
                return self._get_cumulated_result_with_sign(port_name, "electricity", scaling_factor, sign)
