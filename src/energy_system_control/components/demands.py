from energy_system_control.sim.state import SimulationState
from energy_system_control.components.base import Component
from energy_system_control.helpers import *
import os, yaml
import numpy as np
from importlib.resources import files
from typing import List, Dict
from energy_system_control.constants import WATER

class Demand(Component):
    port_name: str
    demand_type: str
    def __init__(self, name: str, demand_type: str):
        self.demand_type = demand_type
        self.port_name = f'{name}_{self.demand_type}_port'
        super().__init__(name, {self.port_name: self.demand_type})


class ConstantPowerDemand(Demand):
    def __init__(self, name: str, demand_type: str, power: float):
        super().__init__(name, demand_type)
        self.power = power  # Since it is a demand, the power is always negative
    
    def step(self, state: SimulationState, action = None): 
        self.ports[self.port_name].flow[self.demand_type] = self.power * state.time_step


class HotWaterDemand(Demand):
    def __init__(self, name: str, reference_temperature: float):
        self.demand_type = 'fluid'
        self.T_ref = C2K(reference_temperature)
        super().__init__(name, self.demand_type)
        

    def resample_data(self, time_step: float, sim_end: float):
        # Resamples the raw data to the format required 
        if hasattr(self, 'raw_data'):
            target_freq = f"{round(time_step/60)}min"
            self.data = resample_with_interpolation(self.raw_data, target_freq, sim_end, var_type="extensive")

    def step(self, state: SimulationState, action = None):
        T_cold_water = state.environmental_data['Temperature cold water']
        T_hot_water = self.ports[self.port_name].T 
        temp = self.data[state.time_id] / state.time_step * 3600  # This calculates the required power in kW (note: time step is in [s], read value in [kWh], hence the 3600)
        mdot_dhw_th = temp / WATER.cp / (313.25 - T_cold_water)  # Theroetical hot water mass flow, in kg/s
        if T_hot_water > self.T_ref:
            mdot = mdot_dhw_th * (313.25 - T_cold_water) / (T_hot_water - T_cold_water)  # Actual hot water mass flow, in kg/s
        else:
            mdot = mdot_dhw_th
        Qdot = mdot * WATER.cp * T_hot_water  # Enthalpy flow output, in kW
        # Remember: flows are POSITIVE if they ENTER the component
        self.ports[self.port_name].flow['heat'] = Qdot * state.time_step
        self.ports[self.port_name].flow['mass'] = mdot * state.time_step


class IEAHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, reference_temperature: float, profile_name: str):
        super().__init__(name, reference_temperature)
        path = files("energy_system_control.data") / "dhw_profiles_iea.csv"
        self.raw_data = pd.read_csv(path, sep = ";", decimal = '.', index_col = 0, header = 0, parse_dates = True, date_format='%H:%M')[profile_name]


class CustomProfileHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, reference_temperature: float, data_path: str, filename:str):
        super().__init__(name, reference_temperature)
        self.raw_data = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', parse_dates = True)