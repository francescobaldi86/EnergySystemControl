from energy_system_control.core.base_classes import Component, Node
from energy_system_control.helpers import *
import os, yaml
import numpy as np
from importlib.resources import files


class Demand(Component):
    def __init__(self, name: str, connections: list):
        super().__init__(name, connections)


class ThermalLoss(Demand):
    def __init__(self, name: str, node: str, T_ambient: float, U_W_per_K: float):
        super().__init__(name, nodes = [node])
        self.T_amb = T_ambient
        self.U = U_W_per_K

    def step(self, dt_s):
        T_node = self.connections.T
        Q_W = -self.U * (T_node - self.T_amb)  # negative if node hotter than ambient
        return {self.node: Q_W * dt_s}
    
class HotWaterDemand(Demand):
    thermal_node: str
    mass_node: str
    def __init__(self, name: str, thermal_node: str, mass_node: str, reference_temperature: float):
        super().__init__(name, [thermal_node, mass_node])
        self.thermal_node = thermal_node
        self.mass_node = mass_node
        self.T_ref = C2K(reference_temperature)

    def resample_data(self, time_step: float, sim_end: float):
        # Resamples the raw data to the format required 
        if hasattr(self, 'raw_data'):
            target_freq = f"{round(time_step/60)}T"
            self.data = resample_with_interpolation(self.raw_data, target_freq, sim_end, var_type="extensive")

    def step(self, time_step: float, environmental_data: dict, action = None):
        T_cold_water = environmental_data['Temperature cold water']
        T_hot_water = self.nodes[self.thermal_node].T 
        temp = self.data[self.time_id] / time_step * 3600  # This calculates the required power in kW (note: time step is in [s], read value in [kWh], hence the 3600)
        if temp > 0.0:
            pass
        mdot_dhw_th = temp / 4.187 / (313.25 - T_cold_water)  # Theroetical hot water mass flow, in kg/s
        if T_hot_water > self.T_ref:
            mdot = mdot_dhw_th * (313.25 - T_cold_water) / (T_hot_water - T_cold_water)  # Actual hot water mass flow, in kg/s
        else:
            mdot = mdot_dhw_th
        Qdot = mdot * 4.187 * T_hot_water  # Enthalpy flow output, in kW
        return {self.thermal_node: -Qdot * time_step, self.mass_node: -mdot * time_step}  # output in {kJ, kg} 


class IEAHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, thermal_node: str, mass_node:str, reference_temperature: float, profile_name: str):
        super().__init__(name, thermal_node, mass_node, reference_temperature)
        path = files("energy_system_control.data") / "dhw_profiles_iea.csv"
        self.raw_data = pd.read_csv(path, sep = ";", decimal = '.', index_col = 0, header = 0, parse_dates = True)[profile_name]


class CustomProfileHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, node: str, reference_temperature: float, data_path: str, filename:str):
        super().__init__(name, node, reference_temperature)
        self.raw_data = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', parse_dates = True)