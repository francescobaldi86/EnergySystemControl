from energy_system_control.components.base import Component, TimeSeriesData
from energy_system_control.helpers import *
from energy_system_control.sim.state import SimulationState
from typing import List, Dict
import os, yaml, csv, json, requests
import numpy as np
import pandas as pd
from typing import Literal


class Producer(Component):
    port_name: str
    production_type: str
    def __init__(self, name: str, production_type: str):
        self.production_type = production_type
        self.port_name = f'{name}_{self.production_type}_port'
        super().__init__(name, {self.port_name: self.production_type})


class ConstantPowerProducer(Producer):
    def __init__(self, name: str, production_type: str, power: float):
        super().__init__(name, production_type)
        self.power = power  # 
    
    def step(self, state: SimulationState, action = None): 
        self.ports[self.port_name].flow[self.production_type] = -self.power * state.time_step  # Since it is a producer, the net energy flow is always negative


class PVpanel(Producer):
    ts: TimeSeriesData
    def __init__(self, name: str, ts: TimeSeriesData):
        super().__init__(name, 'electricity')
        self.ts = ts

    def step(self, state: SimulationState, action = None):
        self.ports[self.port_name].flow['electricity'] = -self.ts.data[state.time_id] * state.time_step


class PVpanelFromData(PVpanel):
    def __init__(self, name: str, data_path: str, filename: str, column_name: str = 'P', date_format: str = '%Y%m%d:%H%M', skipfooter: int = 0, var_unit: Literal['kW', 'W'] = 'W', rescale_factor: float | None = None):
        temp = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', skipfooter = skipfooter, engine='python', index_col = 0)
        temp['time'] = pd.to_datetime(raw_data.index, format=date_format, utc=True)
        temp = raw_data.set_index('time')
        temp = raw_data[column_name]
        if rescale_factor:
            raw_data *= rescale_factor
        ts = TimeSeriesData(
            raw = temp,
            var_type = 'power',
            var_unit = var_unit,
        )
        super().__init__(name, ts)


class PVpanelFromPVGISData(PVpanelFromData):
    def __init__(self, name: str, installed_power: float, data_path: str, filename: str):
        super().__init__(name, installed_power, data_path, filename, column_name = 'P', date_format = '%Y%m%d:%H%M', skipfooter=11, unit = 'W')


class PVpanelFromPVGIS(PVpanel):
    def __init__(self, name: str, installed_power: float, latitude: float, longitude: float, tilt: float, azimuth: float, loss: float = 14, years: list[int] = [2023]):
        """
        Reads data from PVGIS for the selected location. 

        Parameters
        ----------
        latitude : float
         	Latitude [deg], south is negative.
        longitude : float
           	Longitude [deg], west is negative.
        tilt : float
            Inclination angle [deg] from horizontal plane of the (fixed) PV system. ("angle" on PVGIS)
        azimuth : float
            Orientation (azimuth) angle [deg] of the (fixed) PV system, 0=south, 90=west, -90=east. ("aspect" on PVGIS)
        loss: float
            System losses [%] of the raw electric power generated. Defaults to 14.
        years: list[int]
            Years to be loaded from PVGis. Defaults to [2023]
        """
        self.latitude = latitude
        self.longitude = longitude
        self.tilt = tilt
        self.azimuth = azimuth
        self.loss = loss
        self.years = years
        self.installed_power = installed_power
        ts = TimeSeriesData(
            raw = self.pvgis_api_call(),
            var_type = 'power',
            unit = 'W'
        )
        super().__init__(name, installed_power, ts)

    def pvgis_api_call(self):
        url_base = f"https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?"
        pvgis_params = dict(
            lat = self.latitude,
            lon = self.longitude,
            peakpower = self.installed_power,
            loss = self.loss,
            angle = self.tilt,
            azimuth = self.azimuth,
            startyear = self.years[0],
            endyear = self.years[-1],
            pvcalculation = 1,
            outputformat = 'json')
        params = "&".join([f'{key}={value}' for key, value in pvgis_params.items()])
        url_pvcalc = f'{url_base}&{params}'
        temp = pd.DataFrame(requests.get(url_pvcalc).json()['outputs']['hourly'])
        temp['time'] = pd.to_datetime(temp['time'], format="%Y%m%d:%H%M", utc=True)
        temp = temp.set_index('time')
        return temp['P']
        # row_json = json.loads(response.text)   
        #https://re.jrc.ec.europa.eu/api/PVcalc?lat=45&lon=8&peakpower=1&loss=14
