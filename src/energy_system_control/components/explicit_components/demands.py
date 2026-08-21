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
from datetime import datetime

DemandType = Literal['electricity', 'fluid']
VariableType = Literal['energy', 'power', 'volume', 'mass', 'temperature']
VariableUnit = Literal['Wh', 'kWh', 'MWh', 'W', 'kW', 'MW', 'l', 'm3', 'kg', 'C', 'K']

class Demand(ExplicitComponent):
    port_name: str
    demand_type: DemandType
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
        self.ports[self.port_name].flow[self.demand_type] = self._apply_uncertainty(self.power)


class TimeSeriesDemand(Demand):

    demand_type: DemandType | None = None

    def __init__(self, 
                 name: str, 
                 ts_data: TimeSeriesData, 
                 demand_type: str | None = None, 
                 rescale_factor: float = 1.0, 
                 uncertainty_model: UncertaintyModel = NoUncertainty, 
                 uncertainty_seed: int | None = None):
        self.ts = ts_data
        self.rescale_factor = rescale_factor
        if demand_type is None:
            demand_type = self.demand_type  # This can be specified when a sublcass is used
        if demand_type is None:
            raise ValueError("demand_type must be specified for a TimeSeriesDemand object")
        super().__init__(name, demand_type, uncertainty_model, uncertainty_seed)

    @classmethod
    def from_csv(cls,
                    name: str,
                    demand_type: str | None = None,
                    time_alignment: TimeAlignment | None = None,
                    path: str | None = None,
                    column_name: str | None = None,
                    var_type: VariableType = "energy", 
                    var_unit: VariableUnit = 'kWh',
                    rescale_factor: float = 1.0,
                    date_format: str | None = None,
                    **kwargs
                    ):
        """
        Create an TimeSeriesDemand object from a CSV file.

        Parameters
        ----------
        path : str
            Path to the csv_file
        column_name: str
            Name of the column containing the data related to the electricity demand
        var_type: str, optional
            Type of the variable. Default is "energy".
        var_unit: str, optional
            Unit of the variable. Should be compatible with the input provided for var_type. Default is "kWh".
        rescale_factor: float, optional
            Factor to rescale the data. Default is 1.0. 
        **kwargs
            Additional arguments passed to TimeSeriesDemand.

        Returns
        -------
        TimeSeriesDemand object
        """
        csvread_kwargs = {}
        if date_format:
            csvread_kwargs["date_format"] = date_format
        ts_data = TimeSeriesData(
                    raw = pd.read_csv(path, sep = ";", decimal = '.', index_col = 0, header = 0, parse_dates = True, **csvread_kwargs)[column_name],
                    time_alignment = time_alignment,
                    var_type = var_type,
                    var_unit = var_unit)
        return cls(name = name, 
                    ts_data = ts_data,
                    demand_type = demand_type, 
                    rescale_factor = rescale_factor,
                    **kwargs)

    @classmethod
    def from_dataframe(cls,
                    name: str,
                    df: pd.DataFrame | pd.Series, 
                    time_alignment: TimeAlignment,
                    demand_type: str | None = None,
                    column_name: str | None = None,
                    var_type: VariableType = "energy", 
                    var_unit: VariableUnit = 'kWh',
                    rescale_factor: float = 1.0,
                    **kwargs
                    ):
        """
        Create an ElectricityDemand from a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas Dataframe containing the data 
        colummn_name: str
            Name of the column of the dataframe containing the data. Not required if data is a series.
        var_type: str
            Type of the variable. Default is "energy".
        var_unit: str
            Unit of the variable. Should be compatible with the input provided for var_type. Default is "kWh".
        rescale_factor: float
            Factor to rescale the data. Default is 1.0. 
        **kwargs
            Additional arguments passed to TimeSeriesDemand.

        Returns
        -------
        TimeSeriesDemand
        """
        if isinstance(df, pd.DataFrame):
            raw = df[column_name]
        elif isinstance(df, pd.Series):
            raw = df
        else:
            raise TypeError('The input "df" provided should be either a pandas Series or DataFrame.')
        ts_data = TimeSeriesData(
                    raw = raw,
                    time_alignment = time_alignment,
                    var_type = var_type,
                    var_unit = var_unit)
        return cls(name = name, 
                    ts_data = ts_data,
                    demand_type = demand_type,  
                    rescale_factor = rescale_factor,
                    **kwargs)
    

    def resample_data(self, time_step_h: float, simulation_end_h: float, simulation_start_datetime: datetime):
        self.ts.resample(
            time_step_h=time_step_h, 
            simulation_end_h=simulation_end_h, 
            simulation_start_datetime=simulation_start_datetime)
        self.ts.data = self.ts.data * self.rescale_factor
    

class ElectricityDemand(TimeSeriesDemand):
    """
        Electricity demand component based on a time series.
    
        The class can be instantiated directly from an already prepared
        time series, or constructed using one of the alternative constructors:
    
            ElectricityDemand(ts_data)
            ElectricityDemand.from_csv(path)
            ElectricityDemand.from_dataframe(dataframe)
    
        Parameters
        ----------
        data : TimeSeriesData
            Electricity demand time series.
        **kwargs
            Additional arguments passed to TimeSeriesDemand.
    """

    demand_type = "electricity"

    def __init__(
        self,
        name: str,
        ts_data: TimeSeriesData,
        rescale_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            ts_data=ts_data,
            rescale_factor=rescale_factor,
            **kwargs,
        )

    def step(self, state: SimulationState, action = None):
        temp_kW = self.ts.data[state.time_id]  # This calculates the required power in kW (note: time step is in [s], read value in [kWh], hence the 3600)
        self.ports[self.port_name].flows['electricity'] = temp_kW  # Value in kJ


class HotWaterDemand(TimeSeriesDemand):
    """
        Hot water demand component based on a time series.
    
        The class can be instantiated directly from an already prepared
        time series, or constructed using one of the alternative constructors:
    
            HotWaterDemand(name, ts_data)
            HotWaterDemand.from_csv(name, data_path, filname, column_name)
            HotWaterDemand.from_dataframe(name, dataframe, column_name)
            HotWaterDemand.from_iea(name, profile_name)
    
        Parameters
        ----------
        ts_data : TimeSeriesData
            Electricity demand time series.
        **kwargs
            Additional arguments passed to TimeSeriesDemand.
    """

    demand_type = "fluid"

    def __init__(
        self,
        name: str,
        ts_data: TimeSeriesData,
        reference_temperature: float = 40,
        rescale_factor: float = 1.0,
        **kwargs,
    ):
        self.T_ref = C2K(reference_temperature)

        super().__init__(
            name=name,
            ts_data=ts_data,
            rescale_factor=rescale_factor,
            **kwargs,
        )

    def step(self, state: SimulationState, action = None):
        T_cold_water = state.environmental_data.temperature_cold_water
        T_hot_water = self.ports[self.port_name].T 
        demand_kW = self.ts.data[state.time_id]  # This calculates the required power in kW (note: time step is in [s], read value in [kWh], hence the 3600)
        if demand_kW > 0:
            pass
        mdot_dhw_th = demand_kW / WATER.cp / (self.T_ref - T_cold_water)  # Theroetical hot water mass flow, in kg/s
        if T_hot_water > self.T_ref:
            mdot = mdot_dhw_th * (self.T_ref - T_cold_water) / (T_hot_water - T_cold_water)  # Actual hot water mass flow, in kg/s
        else:
            mdot = mdot_dhw_th
        Qdot = mdot * WATER.cp * T_hot_water  # Enthalpy flow output, in kW
        # Remember: flows are POSITIVE if they ENTER the component
        self.ports[self.port_name].flows['heat'] = Qdot
        self.ports[self.port_name].flows['mass'] = mdot
    
    @classmethod
    def from_iea(
        cls,
        name: str,
        profile_name: str,
        reference_temperature: float = 40,
        var_unit: str = "kWh",
        rescale_factor: float = 1.0,
        **kwargs,
    ):
        path = files(
            "energy_system_control.data"
        ) / "dhw_profiles_iea.csv"

        return cls.from_csv(
            name=name,
            time_alignment='daily',
            path = path,
            column_name=profile_name,
            reference_temperature=reference_temperature,
            var_unit=var_unit,
            rescale_factor=rescale_factor,
            **kwargs,
        )