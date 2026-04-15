from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.components.storage_units.electric_storage import BatteryPack, LithiumIonBatteryPack

class BatteryCharger(ControlledComponent):

    def __init__(self, name, 
                 main_port_name: str, 
                 max_charging_power: float, 
                 max_discharging_power: float,
                 battery_pack: BatteryPack, 
                 efficiency_charge: float = 0.95, 
                 efficiency_discharge: float = 0.97):
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.max_charging_power = max_charging_power
        self.max_discharging_power = max_discharging_power
        self.internal_port_name = f'{name}_port'
        self.main_port_name = main_port_name
        self.battery_pack = battery_pack
        super().__init__(name = name,
                         ports_info = {self.internal_port_name: 'electricity', self.main_port_name: 'electricity'})
        
    def step(self, state: SimulationState, action):
        if action >= 0:
            charging_power = min(action, self.get_maximum_charge_power())
            self.ports[self.main_port_name].flows['electricity'] = charging_power
            self.ports[self.internal_port_name].flows['electricity'] = -charging_power * self.efficiency_charge
        else:
            discharging_power = max(action, -self.get_maximum_discharge_power())
            self.ports[self.main_port_name].flows['electricity'] = discharging_power
            self.ports[self.internal_port_name].flows['electricity'] = -discharging_power / self.efficiency_discharge

    def get_maximum_charge_power(self):
        # In a lithium-ion battery, we assume that the maximum charging power is constant between SOC_min and SOC_max, and then decreases linearly to zero between SOC_max and 1.0
        if isinstance(self.battery_pack, LithiumIonBatteryPack):
            SOC = self.battery_pack.SOC
            SOC_max = self.battery_pack.SOC_max 
            if SOC < SOC_max:
                return self.max_charging_power
            else:
                return self.max_charging_power * (1.0 - SOC) / (1.0 - SOC_max)
        else:  # If it's not lithium-ion, we assume constant max charging power
            return self.max_charging_power
    
    def get_maximum_discharge_power(self):
        # In a lithium-ion battery, we assume that the maximum discharging power is constant between SOC_min and SOC_max, and then decreases linearly to zero between SOC_min and 0.0
        if isinstance(self.battery_pack, LithiumIonBatteryPack):
            SOC = self.battery_pack.SOC
            SOC_min = self.battery_pack.SOC_min
            if SOC > SOC_min:
                return self.max_discharging_power
            else:
                return self.max_discharging_power * SOC / SOC_min
        else:  # If it's not lithium-ion, we assume constant max discharging power
            return self.max_discharging_power
        