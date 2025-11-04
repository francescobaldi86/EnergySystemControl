from energy_system_control.core.base_classes import Component
from energy_system_control.helpers import *
from typing import List
import os, yaml, csv, json, requests
import numpy as np
import pandas as pd


class Producer(Component):
    def __init__(self, name: str, nodes: List[str]):
        super().__init__(name, nodes)

class PVpanel(Producer):
    electrical_node: str
    installed_power: float
    raw_data: pd.Series
    def __init__(self, name: str, electrical_node: str, installed_power: float):
        super().__init__(name, [electrical_node])
        self.installed_power = installed_power

    def resample_data(self, time_step: float, sim_end: float):
        # Resamples the raw data to the format required 
        if hasattr(self, 'raw_data'):
            target_freq = f"{round(time_step/60)}T"
            self.data = resample_with_interpolation(self.raw_data, target_freq, sim_end, var_type="intensive")

    def step(self, time_step: float, nodes: list, environmental_data: dict, action = None):
        temp = self.data[self.time_id] * self.installed_power  # The raw data is expected in terms of capacity factor (that is, adimensional)
        if temp > 0:
            pass
        return {self.nodes[0]: temp * time_step}  # Output is in kJ, so kW * h * 
    
    def check_data(self):
        if self.raw_data.between(0, 1).sum() != len(self.raw_data):
            raise ValueError(f'The data for the capacity factor of the PV {self.name} should be between 0 and 1. Please check it')

class PVpanelFromData(PVpanel):
    def __init__(self, name: str, electrical_node: str, installed_power: float, data_path: str, filename: str, datetime_format: str = '%Y%m%D:%H%M'):
        super().__init__(name, electrical_node, installed_power)
        self.raw_data = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', parse_dates = True)
        self.check_data()

class PVpanelFromPVGIS(PVpanel):
    def __init__(self, name: str, electrical_node: str, installed_power: float, latitude: float, longitude: float, tilt: float, azimuth: float, loss: float = 14, years: list[int] = [2023]):
        """
        Reads data from PVGIS for the selected location. 
        :param: latitude    	Latitude, in decimal degrees, south is negative.
        :param: longitude      	Longitude, in decimal degrees, west is negative.
        :param: loss        	Sum of system losses, in percent.
        :param: tilt            Inclination angle from horizontal plane of the (fixed) PV system. ("angle" on PVGIS)
        :param: azimuth         Orientation (azimuth) angle of the (fixed) PV system, 0=south, 90=west, -90=east. ("aspect" on PVGIS)
        """
        super().__init__(name, electrical_node, installed_power)
        self.latitude = latitude
        self.longitude = longitude
        self.tilt = tilt
        self.azimuth = azimuth
        self.loss = loss
        self.years = years
        self.pvgis_api_call()
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
        self.raw_data = temp['P'] / 1000
        # row_json = json.loads(response.text)   
        #https://re.jrc.ec.europa.eu/api/PVcalc?lat=45&lon=8&peakpower=1&loss=14
