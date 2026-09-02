from energy_system_control.constants import WATER
from energy_system_control.components.base import ImplicitComponent
from energy_system_control.helpers import *
from energy_system_control.sim.state import SimulationState
import os, requests
import numpy as np
import pandas as pd
from typing import Literal


class SolarCollector(ImplicitComponent):
    """
    Solar thermal panel model that calculates power output from solar irradiation.
    """
    input_port_name: str
    output_port_name: str
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float, efficiency_0, efficiency_1):
        """
        Parameters
        ----------
        name : str
            Name of the PV panel
        tilt : float
            Tilt angle of the panel in degrees (0 = horizontal)
        azimuth : float
            Azimuth angle in degrees (0 = south, 90 = west, -90 = east)
        surface_area : float
            Installed nominal power of the panel at standard test conditions (kW)
        latitude: float
            Latitude of the location in degrees
        longitude: float
            Longitude of the location in degrees
        """
        self.input_port_name = f'{name}_water_input_port'
        self.output_port_name = f'{name}_water_outlet_port'
        super().__init__(name, ports_info = {
            self.input_port_name: 'fluid',
            self.output_port_name: 'fluid'
        })  # no external time series
        self.tilt_deg = tilt
        self.azimuth_deg = azimuth
        self.tilt_rad = np.radians(tilt)
        self.azimuth_rad = np.radians(azimuth)
        self.surface_area = surface_area
        self.latitude = latitude
        self.longitude = longitude
        self.efficiency_0 = efficiency_0
        self.efficiency_1 = efficiency_1

    @property
    def tilt(self, unit: str = 'deg') -> float:
        match unit:
            case 'deg':
                return self.tilt_deg
            case 'rad':
                return self.tilt_rad
            case _:
                raise ValueError(f'The unit for the tilt should be either rad or deg. {unit} was provided.')

    def step(self, state: SimulationState, action=None):
        env_data = state.environmental_data
        current_datetime = state.simulation_start_datetime + pd.Timedelta(seconds=state.time)
        solar_zenith, solar_azimuth = calculate_solar_angles(self.latitude, self.longitude, current_datetime)
        # Incidence angle
        poa_irradiation = calculate_effective_irradiance(solar_zenith, solar_azimuth, self.tilt_rad, self.azimuth_rad, env_data.direct_irradiation, env_data.diffuse_irradiation)
        efficiency = self.get_efficiency(poa_irradiation, self.ports[self.input_port_name].T, env_data.temperature_ambient)
        mass_flow = self.ports[self.input_port_name].flows['mass']
        Qdot = poa_irradiation * self.surface_area * efficiency if mass_flow > 0.0001 else 0.0
        self.ports[self.output_port_name].flows['mass'] = mass_flow
        self.ports[self.output_port_name].flows['heat'] = self.ports[self.input_port_name].flows['heat'] + Qdot
        self.ports[self.output_port_name].T = (
            self.ports[self.input_port_name].T + Qdot / (mass_flow * WATER.cp)
            if mass_flow > 0.0001
            else self.ports[self.input_port_name].T
        )

    def get_efficiency(self, solar_irradiation: float, inlet_fluid_temperature: float, ambient_air_temperature: float):
        efficiency = self.efficiency_0 + self.efficiency_1 * (inlet_fluid_temperature - ambient_air_temperature) / solar_irradiation
        return efficiency

class UnglazedCollector(SolarCollector):
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float):
        efficiency_0 = (0.8 + 0.9) / 2
        efficiency_1 = (15.0 + 25.0) / 2
        super().__init__(name, tilt, azimuth, surface_area, latitude, longitude, efficiency_0 = efficiency_0, efficiency_1 = efficiency_1)

class FlatPlateCollector(SolarCollector):
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float):
        efficiency_0 = (0.7 + 0.8) / 2
        efficiency_1 = (3.5 + 6.0) / 2
        super().__init__(name, tilt, azimuth, surface_area, latitude, longitude, efficiency_0 = efficiency_0, efficiency_1 = efficiency_1)

class EvacuatedTubeCollector(SolarCollector):
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float):
        efficiency_0 = (0.6 + 0.75) / 2
        efficiency_1 = (1.0 + 2.0) / 2
        super().__init__(name, tilt, azimuth, surface_area, latitude, longitude, efficiency_0 = efficiency_0, efficiency_1 = efficiency_1)
