from energy_system_control.components.base import StorageUnit
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext

class BatteryPack(StorageUnit):
    port_name: str
    self_discharge_rate: float
    max_capacity: float
    SOC_0: float
    SOC: float
    def __init__(self, name, capacity: float, SOC_0: float = 0.5, self_discharge_rate: float = 0.0):
        """
        Model of a simple electric battery pack

        Parameters
        ----------
        name : str
            Name of the component
        capacity : float
         	Design maximum energy capacity [kWh] of the battery.
        SOC_0 : float, optional
            Battery state of charge [-] at simulation start. Defaults to 0.5
        self_discharge_rate: float, optional
            Battery self discharge rate [-/h], indicating the fraction of the current charge lost per hour. Defaults to 0.0 (no self-discharge)
        """
        self.SOC_0 = SOC_0
        self.SOC = None
        self.max_capacity = capacity * 3600  # Energy capacity, in kJ
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

    def initialize(self, ctx: InitContext):
        self.SOC = self.SOC_0
    
    def step(self, state: SimulationState, action):
        # Just considering charging/discharging losses
        self.check_storage_state()
        self.SOC += self.ports[self.port_name].flows['electricity'] * state.time_step / self.max_capacity
        self.SOC -= self.SOC * self.self_discharge_rate / 3600 * self.max_capacity * state.time_step # The self discharge is input in fraction of current capacity per hour
        
    def get_maximum_charge_power(self):
        return self.max_charging_power
    
    def get_maximum_discharge_power(self):
        return self.max_discharging_power
    

class LithiumIonBatteryPack(BatteryPack):
    SOC_min: float
    SOC_max: float
    def __init__(self, name, capacity: float, SOC_0: float = 0.5, SOC_min: float = 0.3, SOC_max: float = 0.9, self_discharge_rate: float = 0.025/30/24):
        super().__init__(name, capacity, SOC_0, self_discharge_rate)
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max