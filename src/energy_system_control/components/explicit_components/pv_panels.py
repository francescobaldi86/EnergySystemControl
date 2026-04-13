from energy_system_control.components.base import TimeSeriesData
from energy_system_control.components.explicit_components.producers import Producer
from energy_system_control.helpers import *
from energy_system_control.sim.state import SimulationState
import os, requests
import numpy as np
import pandas as pd
from typing import Literal

class PVpanel(Producer):
    ts: TimeSeriesData
    def __init__(self, name: str, ts: TimeSeriesData):
        super().__init__(name, 'electricity')
        self.ts = ts

    def step(self, state: SimulationState, action = None):
        self.ports[self.port_name].flow['electricity'] = -self.ts.data[state.time_id] * state.time_step

    def resample_data(self, time_step_h: float, sim_end_h: float):
        self.ts.resample(time_step_h=time_step_h, sim_end_h=sim_end_h)


class PVpanelFromData(PVpanel):
    """
    PV panel model that uses data from a file to calculate power output.

    Parameters
    ----------
    name : str
        Name of the PV panel
    data_path : str
        Path to the directory containing the data file
    filename : str
        Name of the data file
    column_name : str
        Name of the column in the data file containing the power data. Defaults to 'P'.
    date_format : str
        Format of the date in the data file. Defaults to '%Y%m%d:%H%M'.
    skipfooter : int
        Number of rows to skip at the end of the data file. Defaults to 0.
    var_unit : Literal['kW', 'W']
        Unit of the power data in the data file. Defaults to 'W'.
    rescale_factor : float | None
        Factor to rescale the power data. Defaults to None.
    """
    def __init__(self, name: str, data_path: str, filename: str, column_name: str = 'P', date_format: str = '%Y%m%d:%H%M', skipfooter: int = 0, var_unit: Literal['kW', 'W'] = 'W', rescale_factor: float | None = None):
        temp = pd.read_csv(os.path.join(data_path, filename), sep = ";", decimal = '.', skipfooter = skipfooter, engine='python', index_col = 0)
        temp['time'] = pd.to_datetime(temp.index, format=date_format, utc=True)
        temp = temp.set_index('time')
        temp = temp[column_name]
        if rescale_factor:
            temp *= rescale_factor
        ts = TimeSeriesData(
            raw = temp,
            var_type = 'power',
            var_unit = var_unit,
        )
        super().__init__(name, ts)


class PVpanelFromPVGISData(PVpanelFromData):
    def __init__(self, name: str, data_path: str, filename: str, rescale_factor: float | None = None):
        super().__init__(name, data_path=data_path, filename=filename, column_name = 'P', date_format = '%Y%m%d:%H%M', skipfooter=11, var_unit = 'W', rescale_factor=rescale_factor)


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
            var_unit = 'W'
        )
        super().__init__(name, ts)

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


class PVpanelFromIrradiation(PVpanel):
    """
    PV panel model that calculates power output from solar irradiation.
    """

    def __init__(self, name: str, tilt: float, azimuth: float, installed_power: float):
        """
        Parameters
        ----------
        name : str
            Name of the PV panel
        tilt : float
            Tilt angle of the panel in degrees (0 = horizontal)
        azimuth : float
            Azimuth angle in degrees (0 = south, 90 = west, -90 = east)
        installed_power : float
            Installed nominal power of the panel at standard test conditions (kW)
        """
        super().__init__(name, ts=None)  # no external time series
        self.tilt = tilt
        self.azimuth = azimuth
        self.installed_power = installed_power

    def step(self, state: SimulationState, action=None):
        env_data = state.environmental_data

        # Convert angles to radians
        tilt_rad = np.radians(self.tilt)
        panel_az_rad = np.radians(self.azimuth)
        sun_zenith_rad = np.radians(env_data.solar_zenith)
        sun_az_rad = np.radians(env_data.solar_azimuth)

        # Incidence angle
        cos_theta = (
            np.cos(sun_zenith_rad) * np.cos(tilt_rad) +
            np.sin(sun_zenith_rad) * np.sin(tilt_rad) * np.cos(sun_az_rad - panel_az_rad)
        )
        cos_theta = max(cos_theta, 0)

        # POA irradiance
        poa_irradiation = env_data.direct_irradiation * cos_theta + env_data.diffuse_irradiation * (1 + np.cos(tilt_rad)) / 2

        # AC power
        power_output = poa_irradiation / 1000 * self.installed_power

        # Update PV port
        self.ports[self.port_name].flow['electricity'] = -power_output * state.time_step

