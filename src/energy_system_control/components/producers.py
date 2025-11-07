from energy_system_control.core.base_classes import Component
from energy_system_control.helpers import *
from typing import List, Dict
import os, yaml, csv, json, requests
import numpy as np
import pandas as pd


class Producer(Component):
    def __init__(self, name: str, nodes: List[str]):
        super().__init__(name, nodes)


class ConstantPowerProducer(Producer):
    def __init__(self, name: str, nodes: List[str], power: Dict[str, float]):
        super().__init__(name, nodes)
        self.power = power
    
    def step(self, time_step: float, nodes: dict | None = None, environmental_data: dict | None = None, action = None): 
        return {key: self.power[key] for key in nodes}  # Output is in kJ, but time step is in s

class PVpanel(Producer):
    electrical_node: str
    installed_power: float
    raw_data: pd.Series
    data: pd.Series
    def __init__(self, name: str, electrical_node: str, installed_power: float, raw_data: pd.Series):
        super().__init__(name, [electrical_node])
        self.installed_power = installed_power
        self.raw_data = raw_data

    def resample_data(self, time_step: float, sim_end: float):
        # Resamples the raw data to the format required 
        if hasattr(self, 'raw_data'):
            target_freq = f"{round(time_step/60)}min"
            self.data = resample_with_interpolation(self.raw_data, target_freq, sim_end, var_type="intensive")

    def step(self, time_step: float, environmental_data: dict | None = None, action = None):
        temp = self.data[self.time_id] * self.installed_power  # The raw data is expected in terms of capacity factor (that is, adimensional)
        return {self.nodes[0]: temp * time_step}  # Output is in kJ, but time step is in s
    
    def check_data(self):
        if self.raw_data.between(0, 1).sum() != len(self.raw_data):
            raise ValueError(f'The data for the capacity factor of the PV {self.name} should be between 0 and 1. Please check it')

class PVpanelFromData(PVpanel):
    def __init__(self, name: str, electrical_node: str, installed_power: float, data_path: str, filename: str, datetime_format: str = '%Y%m%D:%H%M'):
        raw_data = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', parse_dates = True)
        super().__init__(name, electrical_node, installed_power, raw_data)
        self.check_data()

class PVpanelFromPVGIS(PVpanel):
    def __init__(self, name: str, electrical_node: str, installed_power: float, latitude: float, longitude: float, tilt: float, azimuth: float, loss: float = 14, years: list[int] = [2023]):
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
        super().__init__(name, electrical_node, installed_power, raw_data=self.pvgis_api_call())
        self.check_data()

    def pvgis_api_call(self):
        url_base = f"https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?"
        pvgis_params = dict(
            lat = self.latitude,
            lon = self.longitude,
            peakpower = 1,
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
        return temp['P'] / 1000
        # row_json = json.loads(response.text)   
        #https://re.jrc.ec.europa.eu/api/PVcalc?lat=45&lon=8&peakpower=1&loss=14
