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
    def __init__(self, SOC_min: float = 0.1, **kwargs):
        # we initialize the mother class creating self.charger and self.battery_pack with the lithium ion type
        super().__init__(**kwargs, battery_type='lithium ion')
        
        self.SOC_min = SOC_min
        
        # saving the original method of the charger to be able to call it later
        original_step = self.charger.step
        
        # we define a new step method that checks the SOC before allowing discharge
        def safe_charger_step(state, action=None):
            # if there is a discharge action and the SOC is below the minimum, we set the action to 0 (no discharge)
            if action is not None and action < 0 and self.battery_pack.SOC <= self.SOC_min:
                action = 0.0  
            return original_step(state, action)
        
        # we override the step method of the charger with our new safe method
        self.charger.step = safe_charger_step

class LeadAcidBattery(Battery):
    def __init__(self, SOC_min: float = 0.3, SOC_derate: float = 0.6, **kwargs):
        """
        Model of a VRLA GEL Deep Cycle Lead-Acid Battery (e.g., Victron BAT412126101).
        
        Parameters
        ----------
        SOC_min : float
            Minimum State of Charge below which the battery stops providing power (Depth of Discharge limit).
        SOC_derate : float
            State of Charge threshold below which the maximum discharge power starts to linearly derate.
        """
        # Set typical physical parameters for GEL VRLA batteries
        kwargs.setdefault('efficiency_charge', 0.85)
        kwargs.setdefault('efficiency_discharge', 0.95)
        # Self-discharge rate of <2% per month (at 20°C) converted to an hourly basis
        kwargs.setdefault('self_discharge_rate', 0.0000277) 
        
        # Initialize the mother class utilizing 'base' to load the standard BatteryPack from BATTERY_REGISTRY
        super().__init__(**kwargs, battery_type='base')
        
        self.SOC_min = SOC_min
        self.SOC_derate = SOC_derate
        
        self.battery_capacity = kwargs.get('capacity', 0.0)
        
        # Save the original step method of the charger to wrap it with the new physical constraints
        original_step = self.charger.step
        
        def safe_charger_step(state, action=None):
            if action is not None: 
                current_soc = self.battery_pack.SOC
                
                # Charge: Overcharge prevention (Virtual Curtailment)
                if action > 0:
                    # Maximum energy (kWh) that the battery can still store before reaching 100%
                    energy_space_kwh = self.battery_capacity * (1.0 - current_soc)
                    
                    # Maximum acceptable power in this step without exceeding SOC = 1.0
                    # This accounts for charging efficiency and time step duration in hours
                    max_charge_power = energy_space_kwh / (self.charger.efficiency_charge * state.time_step)
                    
                    # The final physical limit is the minimum between the remaining thermodynamic space 
                    # and the nominal maximum charging power (C-rate limit)
                    max_charge_power = min(max_charge_power, self.charger.max_charging_power)
                    
                    # Cut the excess power requested by the controller
                    if action > max_charge_power:
                        action = max_charge_power

                # Deep discharge prevention and linear derating 
                elif action < 0:
                    # Absolute cutoff of power output if the SOC drops below the critical minimum threshold
                    if current_soc <= self.SOC_min:
                        action = 0.0  
                    # Linear reduction of the maximum power output when operating in a partial state of charge
                    elif current_soc < self.SOC_derate:
                        derating_factor = (current_soc - self.SOC_min) / (self.SOC_derate - self.SOC_min)
                        max_possible_discharge = -self.charger.max_discharging_power * derating_factor
                        
                        # Limit the requested discharge action to what the battery can physically provide
                        if action < max_possible_discharge:
                            action = max_possible_discharge
                        
            return original_step(state, action)
        
        # Override the default charger step method with the new constrained method
        self.charger.step = safe_charger_step