from energy_system_control.core.base_classes import Component
from energy_system_control.core.nodes import ThermalNode, MassNode, ElectricalStorageNode
from energy_system_control.helpers import *
from typing import Dict, List

class StorageUnit(Component):
    """A lumped hot-water store modeled as node with internal losses."""
    SOC: dict
    state: dict
    max_capacity: dict

    def __init__(self, name: str, nodes: List[str], max_capacity: dict):
        super().__init__(name, nodes)
        self.SOC = {}
        self.max_capacity = max_capacity

    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        return {n: 0.0 for n in self.nodes}
    
    def check_storage_state(self):
        # This function must be implemented for each sub type
        raise(NotImplementedError)
            
        

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
        min_capacity = {n: 0.0 for n in max_capacity.keys()}
        super().__init__(name, nodes, max_capacity)

    def create_storage_nodes(self):
        nodes = {}
        nodes[f'{self.name}_thermal_node'] = ThermalNode(name = f'{self.name}_thermal_node', inertia = self.tank_volume * 4.187, T_0 = self.T_0)
        nodes[f'{self.name}_mass_node'] = MassNode(name = f'{self.name}_mass_node', max_capacity=self.tank_volume, m_0 = self.tank_volume)
        return nodes
        
class Battery(StorageUnit):
    crate: float
    erate: float
    efficiency_charge: float
    efficiency_discharge: float
    def __init__(self, name, capacity: float, electrical_node: str | None = None, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5):
        self.crate = crate
        self.erate = erate
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.SOC_0 = SOC_0
        self.max_capacity = capacity * 3600  # Energy capacity, in kJ
        self.electrical_node = electrical_node if electrical_node else f'{name}_electrical_node'
        super().__init__(name, [self.electrical_node], {self.electrical_node: self.max_capacity})

    def create_storage_nodes(self):
        nodes = {}
        if self.electrical_node == f'{self.name}_electrical_node':
            nodes[f'{self.name}_electrical_node'] = ElectricalStorageNode(name = self.electrical_node, max_capacity = self.max_capacity[self.electrical_node], SOC_0 = self.SOC_0)
        # self.verify_connected_components()
        return nodes
    
    def check_storage_state(self, nodes):
        if nodes[self.nodes[0]].state_variable > self.max_capacity * self.SOC_max:
            raise(StorageError, f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.time}')
        elif nodes[self.nodes[0]].state_variable < self.max_capacity * self.SOC_min:
            raise(StorageError, f'Storage unit {self.name} has storage level lower than zero at time {self.time}')
    
    def verify_connected_components(self):
        raise(NotImplementedError)
    
    def step(self, time_step: float, nodes: list, environmental_data: dict, action):
        # Just considering charging/discharging losses
        if nodes[self.nodes[0]].delta > 0: 
            return {self.nodes[0]: -nodes[self.nodes[0]].delta * (1 - self.efficiency_charge)}
        else:
            return {self.nodes[0]: nodes[self.nodes[0]].delta * (1 - self.efficiency_discharge)}