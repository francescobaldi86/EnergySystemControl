from energy_system_control.core.base_classes import Component
from energy_system_control.core.nodes import ThermalNode, MassNode, ElectricalStorageNode
from energy_system_control.helpers import *
from typing import Dict, List
import warnings

class StorageUnit(Component):
    """A lumped hot-water store modeled as node with internal losses."""
    state: dict
    max_capacity: dict

    def __init__(self, name: str, nodes: List[str], max_capacity: Dict[str, float]):
        super().__init__(name, nodes)
        self.max_capacity = max_capacity

    def step(self, action):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        raise(NotImplementedError)
    
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
    
    def step(self, action):
        output = {}
        for node_name, node in self.nodes.items():
            output[node_name] = 0.0
        return output
        
class Battery(StorageUnit):
    crate: float
    erate: float
    efficiency_charge: float
    efficiency_discharge: float
    def __init__(self, name, capacity: float, electrical_node: str | None = None, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5):
        """
        Model of a simple electric battery

        Parameters
        ----------
        name : str
            Name of the component
        capacity : float
         	Design maximum energy capacity [kWh] of the battery.
        electrical_node : float, optional
           	Name of the electrical node of the battery. If not provided, it will automatically be assigned to f"{name}_electrical_node"
        crate : float, optional
            C-rate [-] of the battery. Sets the maximum charge power based on the capacity (if crate = 1, P_charge_max [kW] = E_max [kWh]) Defaults to 1.
        erate : float, optional
            E-rate [-] of the battery. Sets the maximum discharge power based on the capacity (if erate = 1, P_discharge_max [kW] = E_max [kWh]) Defaults to 1.
        efficiency_charge : float, optional
            Efficiency [-] of the charging process of the battery. Defaults to 0.92 [REF]
        efficiency_discharge : float, optional
            Efficiency [-] of the discharging process of the battery. Defaults to 0.94 [REF]
        SOC_0 : float, optional
            Battery state of charge [-] at simulation start. Defaults to 0.5
        """
        self.crate = crate
        self.erate = erate
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.SOC_0 = SOC_0
        self.max_battery_capacity = capacity * 3600  # Energy capacity, in kJ
        self.max_charging_power = capacity * self.crate
        self.max_discharging_power = capacity * self.erate
        self.io_node_name = electrical_node
        self.internal_node_name = f'{name}_electrical_node'
        super().__init__(name, [self.internal_node_name, self.io_node_name], {self.internal_node_name: self.max_battery_capacity})
    
    @property
    def SOC(self): 
        return self.nodes[self.node_names[0]].SOC

    def create_storage_nodes(self):
        nodes = {}
        nodes[f'{self.name}_electrical_node'] = ElectricalStorageNode(name = self.internal_node_name, max_capacity = self.max_capacity[self.internal_node_name], SOC_0 = self.SOC_0)
        # self.verify_connected_components()
        return nodes
    
    def check_storage_state(self):
        if self.SOC > 1.0:
            warnings.warn(f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.time}. Observed SOC is {self.SOC} while max value is 1.0', UserWarning)
        elif self.SOC < 0.0:
            warnings.warn(f'Storage unit {self.name} has storage level lower than minimum allowed at time step {self.time}. Observed SOC is {self.SOC} while min value is 0.0', UserWarning)
    
    def verify_connected_components(self):
        raise(NotImplementedError)
    
    def step(self, action):
        # Just considering charging/discharging losses
        self.check_storage_state()
        if action > 0: 
            return {self.internal_node_name: action * self.efficiency_charge,
                    self.io_node_name: -action}
        else:
            return {self.internal_node_name: action * self.efficiency_discharge,
                    self.io_node_name: -action}
        
    def get_maximum_charge_power(self):
        raise NotImplementedError
    
    def get_maximum_discharge_power(self):
        raise NotImplementedError
    

class LithiumIonBattery(Battery):
    SOC_min: float
    SOC_max: float
    def __init__(self, name, capacity: float, electrical_node: str | None = None, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5, SOC_min: float = 0.3, SOC_max: float = 0.9):
        super().__init__(name, capacity, electrical_node, crate, erate, efficiency_charge, efficiency_discharge, SOC_0)
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max

    def get_maximum_charge_power(self):
        # In a lithium-ion battery, we assume that the maximum charging power is constant between SOC_min and SOC_max, and then decreases linearly to zero between SOC_max and 1.0
        if self.SOC < self.SOC_max:
            return self.max_charging_power
        else:
            return self.max_charging_power * (1.0 - self.SOC) / (1.0 - self.SOC_max)
    
    def get_maximum_discharge_power(self):
        # In a lithium-ion battery, we assume that the maximum discharging power is constant between SOC_min and SOC_max, and then decreases linearly to zero between SOC_min and 0.0
        if self.SOC > self.SOC_min:
            return self.max_discharging_power
        else:
            return self.max_discharging_power * self.SOC / self.SOC_min