from EnergySystemControl.environments.base_environment import Component
from EnergySystemControl.helpers import read_timeseries_data_to_numpy, C2K, K2C
import os
import numpy as np
import yaml


class Demand(Component):
    def __init__(self, name: str, node: str, project_path: str):
        super().__init__(self, name, project_path)


class ThermalLoss(Demand):
    def __init__(self, name: str, node: str, T_ambient: float, U_W_per_K: float, project_path: str):
        super().__init__(name, node, project_path)
        self.node = node
        self.T_amb = T_ambient
        self.U = U_W_per_K

    def step(self, dt_s, nodes):
        T_node = nodes[self.node].T
        Q_W = -self.U * (T_node - self.T_amb)  # negative if node hotter than ambient
        return {self.node: Q_W * dt_s}
    
class HotWaterDemand(Demand):
    def __init__(self, name: str, thermal_node: str, mass_node: str, reference_temperature: float, project_path: str):
        super().__init__(name, project_path)
        self.thermal_node = thermal_node
        self.mass_node = mass_node
        self.T_ref = reference_temperature

    def step(self, t_s, dt_s, nodes, T_cold_water):
        T_hot_water = nodes[self.thermal_node].T 
        temp = (np.interp(x=(t_s + dt_s) % self.data[0][-1], xp=self.data[0], fp=self.data[1]) -
            np.interp(x=t_s % self.data[0][-1], xp=self.data[0], fp=self.data[1])) / dt_s  # This calculates the required power in kW
        mdot_dhw_th = temp / 4.187 / (C2K(40) - T_cold_water)
        if T_hot_water > self.T_ref:
            mdot = mdot_dhw_th * (C2K(40) - T_cold_water) / (T_hot_water - T_cold_water)
        else:
            mdot = mdot_dhw_th
        Qdot = mdot_dhw_th * 4.187 * T_hot_water
        return {self.thermal_node: Qdot * dt_s, self.mass_node: mdot * dt_s}
    

class IEAHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, node: str, reference_temperature: float, project_path: str, profile_name: str):
        super().__init__(name, node, reference_temperature, project_path)
        self.data = read_timeseries_data_to_numpy(f'{os.path.realpath(__file__)}\\..\\data\\DHW_profiles_IEA.csv', profile_name)

class CustomProfileHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, node: str, reference_temperature: float, project_path: str, filename:str):
        super().__init__(name, node, reference_temperature, project_path)
        self.data = read_timeseries_data_to_numpy(os.path.join(project_path, filename))