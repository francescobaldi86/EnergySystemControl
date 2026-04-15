from energy_system_control.components.base import CompositeComponent
from energy_system_control.components.controlled_components.electric import BatteryCharger
from energy_system_control.components.storage_units.electric_storage import BatteryPack, LithiumIonBatteryPack
from energy_system_control.sim.state import SimulationState

BATTERY_REGISTRY = {
    "base": BatteryPack,
    "lithium ion": LithiumIonBatteryPack,
}

class Battery(CompositeComponent):
    """
    The inverter is a component 
    """
    c_rate: float
    e_rate: float
    main_connection_port_name: str
    charger_port_name: str
    battery_pack_port_name: str
    battery_pack : BatteryPack
    charger: BatteryCharger
    SOC: float

    def __init__(self, 
                 name: str,
                 capacity: float,
                 battery_type: str = 'base', 
                 crate: float = 1.0, 
                 erate: float = 1.0, 
                 efficiency_charge: float = 0.92, 
                 efficiency_discharge: float = 0.94, 
                 SOC_0: float = 0.5, 
                 self_discharge_rate: float = 0.0):
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
        super().__init__(name, {})
        self.crate = crate
        self.erate = erate
        self.main_connection_port_name = f'{name}_electricity_port'
        self.charger_port_name = f'{name}_charger_port'
        self.battery_pack_port_name = f'{name}_pack_electricity_port'
        cls = BATTERY_REGISTRY[battery_type]
        self.battery_pack = cls(name = f'{self.name}_pack', capacity = capacity, SOC_0 = SOC_0, self_discharge_rate = self_discharge_rate)
        self.charger = BatteryCharger(name = f'{name}_charger',
                                      main_port_name = self.main_connection_port_name,
                                      max_charging_power = capacity * self.crate,
                                      max_discharging_power = capacity * self.erate, 
                                      efficiency_charge = efficiency_charge,
                                      efficiency_discharge = efficiency_discharge,
                                      battery_pack = self.battery_pack)

    def get_internal_components(self):
        return {f'{self.name}_pack': self.battery_pack, f'{self.name}_charger': self.charger}
    
    def get_internal_connections(self):
        return [(self.battery_pack_port_name, self.charger_port_name)]
    
    @property
    def SOC(self):
        return self.battery_pack.SOC
    
class LithiumIonBattery(Battery):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, battery_type = 'lithium ion')