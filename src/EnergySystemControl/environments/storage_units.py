from EnergySystemControl.environments.base_environment import Component
from EnergySystemControl.helpers import *
from typing import Dict, List

class StorageUnit(Component):
    """A lumped hot-water store modeled as node with internal losses."""
    storage_variable: str
    SOC: float
    state: float
    max_capacity: float

    def __init__(self, name: str, nodes: List[str], max_capacity: float):
        super().__init__(name)
        self.max_capacity = max_capacity

    def step(self, dt_s, nodes):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        return {}
    
    def check_max_capacity(self):
        if self.state > self.max_capacity:
            raise(MaxStorageError, f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.ts}')
        

class HotWaterStorage(StorageUnit):

    def __init__(self, name, max_capacity):
        super().__init__(self, name, max_capacity)
        