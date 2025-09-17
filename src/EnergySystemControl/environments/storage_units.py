from EnergySystemControl.environments.base_environment import Component
from EnergySystemControl.environments.nodes import *
from EnergySystemControl.helpers import *
from typing import Dict, List

class StorageUnit(Component):
    """A lumped hot-water store modeled as node with internal losses."""
    storage_variables: list
    SOC: dict
    state: dict
    max_capacity: dict

    def __init__(self, name: str, nodes: List[str], max_capacity: dict):
        super().__init__(name, nodes)
        self.max_capacity = max_capacity

    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        return {n: 0.0 for n in self.nodes}
    
    def check_max_capacity(self):
        for var in self.storage_variables:
            if self.state[var] > self.max_capacity[var]:
                raise(MaxStorageError, f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.ts}')
        

class HotWaterStorage(StorageUnit):
    tank_volume: float
    max_temperature: float
    def __init__(self, name, max_temperature: float, tank_volume: float, thermal_node: str | None = None, mass_node: str | None = None, T_0: float = 40.0):
        self.tank_volume = tank_volume
        self.max_temperature = C2K(max_temperature)
        self.T_0 = C2K(T_0)
        if thermal_node and mass_node:
            max_capacity = {
                thermal_node: max_temperature * tank_volume * 4.187,  # Energy capacity, in kJ
                mass_node: tank_volume
            }
            nodes = [thermal_node, mass_node]
        else:
            max_capacity = {
                f'{name}_thermal_node': max_temperature * tank_volume * 4.187,  # Energy capacity, in kJ
                f'{name}_mass_node': tank_volume
            }
            nodes = [f'{name}_thermal_node', f'{name}_mass_node']
        super().__init__(name, nodes, max_capacity)

    def create_storage_nodes(self):
        nodes = {}
        nodes[f'{self.name}_thermal_node'] = ThermalNode(name = f'{self.name}_thermal_node', inertia = self.tank_volume * 4.187, T_0 = self.T_0)
        nodes[f'{self.name}_mass_node'] = MassNode(name = f'{self.name}_mass_node', max_capacity=self.tank_volume, m_0 = self.tank_volume)
        return nodes
        