from energy_system_control.sim.state import SimulationState
from energy_system_control.components.base import ExplicitComponent
from energy_system_control.helpers import *
from energy_system_control.constants import WATER
from energy_system_control.uncertainty import UncertaintyModel, NoUncertainty
from energy_system_control.components.base import TimeSeriesData
import os, yaml
import numpy as np
from importlib.resources import files
from typing import List, Dict, Literal


class Demand(ExplicitComponent):
    port_name: str
    demand_type: str
    uncertainty_model: float
    uncertainty_seed: int
    def __init__(self, name: str, demand_type: str, uncertainty_model: UncertaintyModel = NoUncertainty, uncertainty_seed: int | None = None):
        self.demand_type = demand_type
        self.port_name = f'{name}_{self.demand_type}_port'
        self.uncertainty_model = uncertainty_model
        self.uncertainty_seed = uncertainty_seed
        super().__init__(name, {self.port_name: self.demand_type})

    def initialize(self, ctx) -> None:
        # Create reproducible RNG per component.
        if self.uncertainty_model is not NoUncertainty:
            if not hasattr(ctx, "seed"):
                raise(KeyError, 'The context information should include the seed for the uncertainty model')
            else:
                self.uncertainty_seed = ctx.seed
                self._rng = np.random.default_rng(self.seed)

    def _apply_uncertainty(self, value: float, time_id: int) -> float:
        return self.uncertainty_model.apply(value, rng=self._rng)


class ConstantPowerDemand(Demand):
    def __init__(self, name: str, demand_type: str, power: float, **kwargs):
        super().__init__(name, demand_type, **kwargs)
        self.power = power  # Since it is a demand, the power is always negative
    
    def step(self, state: SimulationState, action = None):
        self.ports[self.port_name].flow[self.demand_type] = self._apply_uncertainty(self.power * state.time_step)


class TimeSeriesDemand(Demand):
    var_type: Literal['energy', 'power', 'volume', 'mass', 'temperature']
    var_unit: Literal['Wh', 'kWh', 'MWh', 'W', 'kW', 'MW', 'l', 'm3', 'kg', 'C', 'K']
    ts: TimeSeriesData
    
    def __init__(self, name: str, demand_type: str, var_type: str, var_unit: str, uncertainty_model: UncertaintyModel = NoUncertainty, uncertainty_seed: int | None = None):
        self.var_type = var_type
        self.var_unit = var_unit
        super().__init__(name, demand_type, uncertainty_model, uncertainty_seed)

    def resample_data(self, time_step_h: float, sim_end_h: float):
        self.ts.resample(time_step_h=time_step_h, sim_end_h=sim_end_h)
    

class ElectricityDemand(TimeSeriesDemand):
    def __init__(self, name: str, **kwargs):
        self.demand_type = 'electricity'
        super().__init__(name, self.demand_type, **kwargs)
        
    def step(self, state: SimulationState, action = None):
        temp_kW = self.ts.data[state.time_id]  # This calculates the required power in kW (note: time step is in [s], read value in [kWh], hence the 3600)
        self.ports[self.port_name].flow['electricity'] = temp_kW * state.time_step  # Value in kJ


class HotWaterDemand(TimeSeriesDemand):
    def __init__(self, name: str, reference_temperature: float, **kwargs):
        self.demand_type = 'fluid'
        self.T_ref = C2K(reference_temperature)
        super().__init__(name, self.demand_type, **kwargs)

    def step(self, state: SimulationState, action = None):
        T_cold_water = state.environmental_data.temperature_cold_water
        T_hot_water = self.ports[self.port_name].T 
        demand_kW = self.ts.data[state.time_id]  # This calculates the required power in kW (note: time step is in [s], read value in [kWh], hence the 3600)
        mdot_dhw_th = demand_kW / WATER.cp / (313.25 - T_cold_water)  # Theroetical hot water mass flow, in kg/s
        if T_hot_water > self.T_ref:
            mdot = mdot_dhw_th * (313.25 - T_cold_water) / (T_hot_water - T_cold_water)  # Actual hot water mass flow, in kg/s
        else:
            mdot = mdot_dhw_th
        Qdot = mdot * WATER.cp * T_hot_water  # Enthalpy flow output, in kW
        # Remember: flows are POSITIVE if they ENTER the component
        self.ports[self.port_name].flow['heat'] = Qdot * state.time_step
        self.ports[self.port_name].flow['mass'] = mdot * state.time_step


class IEAHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, reference_temperature: float, profile_name: str, **kwargs):
        super().__init__(name, reference_temperature, var_type = 'energy', var_unit = 'kWh', **kwargs)
        path = files("energy_system_control.data") / "dhw_profiles_iea.csv"
        self.ts = TimeSeriesData(
            raw = pd.read_csv(path, sep = ";", decimal = '.', index_col = 0, header = 0, parse_dates = True, date_format='%H:%M')[profile_name],
            var_type = 'energy',
            var_unit = 'kWh')

class CustomProfileHotWaterDemand(HotWaterDemand):
    def __init__(self, name: str, reference_temperature: float, data_path: str, filename:str, var_unit: str = 'kWh', **kwargs):
        super().__init__(name, reference_temperature, **kwargs)
        self.ts = TimeSeriesData(
            raw = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', parse_dates = True),
            var_type = 'energy',
            var_unit = var_unit)