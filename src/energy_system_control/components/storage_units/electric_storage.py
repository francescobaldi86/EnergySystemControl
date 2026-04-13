from energy_system_control.components.base import StorageUnit
from energy_system_control.sim.state import SimulationState

class Battery(StorageUnit):
    crate: float
    erate: float
    efficiency_charge: float
    efficiency_discharge: float
    port_name: str
    max_charging_power: float
    max_discharging_power:float
    self_discharge_rate: float
    def __init__(self, name, capacity: float, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5, self_discharge_rate: float = 0.0):
        """
        Model of a simple electric battery

        Parameters
        ----------
        name : str
            Name of the component
        capacity : float
         	Design maximum energy capacity [kWh] of the battery.
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
        self_discharge_rate: float, optional
            Battery self discharge rate [-/h], indicating the fraction of the current charge lost per hour. Defaults to 0.0 (no self-discharge)
        """
        self.crate = crate
        self.erate = erate
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.SOC_0 = SOC_0
        self.max_capacity = capacity * 3600  # Energy capacity, in kJ
        self.max_charging_power = capacity * self.crate
        self.max_discharging_power = capacity * self.erate
        self.self_discharge_rate = self_discharge_rate / 3600  # Provided
        self.port_name = f'{name}_electricity_port'
        super().__init__(name, {self.port_name: 'electricity'})
    
    def check_storage_state(self):
        if self.SOC > 1.0:
            raise ValueError(f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.time}. Observed SOC is {self.SOC} while max value is 1.0')
        elif self.SOC < 0.0:
            raise ValueError(f'Storage unit {self.name} has storage level lower than minimum allowed at time step {self.time}. Observed SOC is {self.SOC} while min value is 0.0')
    
    def verify_connected_components(self):
        raise(NotImplementedError)
    
    def step(self, state: SimulationState, action):
        # Just considering charging/discharging losses
        self.check_storage_state()
        if self.ports[self.port_name].flow['electricity'] > 0:  # Charging
            self.SOC += self.ports[self.port_name].flow['electricity'] * self.efficiency_charge / self.max_capacity
        else:
            self.SOC += self.ports[self.port_name].flow['electricity'] / self.efficiency_discharge / self.max_capacity
        self.SOC -= self.SOC * self.self_discharge_rate * self.max_capacity * state.time_step / 3600 # The self discharge is input in fraction of current capacity per hour
        
    def get_maximum_charge_power(self):
        return self.max_charging_power
    
    def get_maximum_discharge_power(self):
        return self.max_discharging_power
    

class LithiumIonBattery(Battery):
    SOC_min: float
    SOC_max: float
    def __init__(self, name, capacity: float, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5, SOC_min: float = 0.3, SOC_max: float = 0.9, self_discharge_rate: float = 0.025/30/24):
        super().__init__(name, capacity, crate, erate, efficiency_charge, efficiency_discharge, SOC_0, self_discharge_rate)
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