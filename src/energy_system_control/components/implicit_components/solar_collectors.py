from energy_system_control.constants import WATER
from energy_system_control.components.base import ImplicitComponent
from energy_system_control.helpers import *
from energy_system_control.sim.state import SimulationState
import numpy as np
import pandas as pd
from typing import Literal


class SolarCollector(ImplicitComponent):
    """
    Solar thermal panel model that calculates heat output from solar irradiation.
    
    This component models a solar thermal collector that absorbs solar radiation and transfers
    the resulting heat to a fluid (typically water) flowing through the panel. The model calculates
    the plane-of-array (POA) irradiation based on solar angles and weather data, applies thermal
    efficiency calculations, and updates the outlet fluid temperature and heat flow.
    
    Attributes
    ----------
    input_port_name : str
        Name of the fluid input port
    output_port_name : str
        Name of the fluid output port
    """
    input_port_name: str
    output_port_name: str
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float, efficiency_0, efficiency_1):
        """
        Initialize a solar thermal collector.
        
        Parameters
        ----------
        name : str
            Unique identifier for the solar thermal panel
        tilt : float
            Tilt angle of the panel in degrees (0 = horizontal, 90 = vertical)
        azimuth : float
            Azimuth angle in degrees (0 = south, 90 = west, -90 = east, 180 = north)
        surface_area : float
            Collector aperture area in m²
        latitude : float
            Latitude of the location in degrees (positive = North, negative = South)
        longitude : float
            Longitude of the location in degrees (positive = East, negative = West)
        efficiency_0 : float
            Zero-loss collection efficiency (dimensionless), typically 0.6-0.9
        efficiency_1 : float
            Linear efficiency loss coefficient (W/(m²·K)), typically 1.0-25.0
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
        """
        Get the tilt angle of the solar panel.
        
        Parameters
        ----------
        unit : str, optional
            Unit for the returned angle ('deg' for degrees, 'rad' for radians). Default is 'deg'.
        
        Returns
        -------
        float
            The tilt angle in the specified unit
        
        Raises
        ------
        ValueError
            If an unsupported unit is specified
        """
        match unit:
            case 'deg':
                return self.tilt_deg
            case 'rad':
                return self.tilt_rad
            case _:
                raise ValueError(f'The unit for the tilt should be either rad or deg. {unit} was provided.')

    def balance(self, state: SimulationState, action=None):
        """
        Calculate the thermal balance of the solar collector for the current time step.
        
        This method computes:
        1. Solar angles (zenith and azimuth) based on location and time
        2. Plane-of-array (POA) irradiation based on direct and diffuse radiation
        3. Thermal efficiency based on operating temperature difference
        4. Heat output considering fluid flow rate
        5. Outlet fluid temperature and heat flow
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state containing environmental data and time information
        action : optional
            Unused parameter for interface compatibility
        
        Returns
        -------
        bool
            True if the simulation was successful, False if inlet flow rate is None
        list
            List of output port names for which values were updated
        """
        # Calculate the solar angles and effective irradiation
        env_data = state.environmental_data
        current_datetime = state.simulation_start_datetime + pd.Timedelta(seconds=state.time)
        solar_zenith, solar_azimuth = calculate_solar_angles(self.latitude, self.longitude, current_datetime)
        poa_irradiation = calculate_effective_irradiance(solar_zenith, solar_azimuth, self.tilt_rad, self.azimuth_rad, env_data.direct_irradiation, env_data.diffuse_irradiation)
        # Calculate panel efficiency
        efficiency = self.get_efficiency(poa_irradiation, self.ports[self.input_port_name].T, env_data.temperature_ambient)
        # Understand if we are ready or not to simulate the component
        if self.ports[self.input_port_name].flows['mass'] is None:
            return False, []
        mfr = self.ports[self.input_port_name].flows['mass']
        Qdot = poa_irradiation * self.surface_area * efficiency if mfr > 0.0001 else 0.0
        self.ports[self.output_port_name].flows['mass'] = mfr
        self.ports[self.output_port_name].flows['heat'] = self.ports[self.input_port_name].flows['heat'] + Qdot
        self.ports[self.output_port_name].T = (
            self.ports[self.input_port_name].T + Qdot / (mfr * WATER.cp)
            if mfr > 0.0001
            else self.ports[self.input_port_name].T
        )
        return True, [self.output_port_name]

    def get_efficiency(self, solar_irradiation: float, inlet_fluid_temperature: float, ambient_air_temperature: float):
        """
        Calculate the thermal efficiency of the solar collector.
        
        Uses the Hottel-Whillier-Bliss equation for flat plate solar collectors:
        η = η₀ - η₁ * (T_in - T_amb) / G
        
        Parameters
        ----------
        solar_irradiation : float
            Plane-of-array solar irradiation in W/m²
        inlet_fluid_temperature : float
            Temperature of the fluid entering the collector in K or °C
        ambient_air_temperature : float
            Ambient air temperature in K or °C
        
        Returns
        -------
        float
            Thermal efficiency of the collector (dimensionless, typically 0.0-0.9)
        """
        efficiency = self.efficiency_0 + self.efficiency_1 * (inlet_fluid_temperature - ambient_air_temperature) / solar_irradiation
        return efficiency

class UnglazedCollector(SolarCollector):
    """
    Unglazed solar thermal collector (swimming pool collector).
    
    Unglazed collectors have no transparent cover and are used primarily for low-temperature
    applications such as pool heating. They have high zero-loss efficiency (0.8-0.9) but also
    higher heat loss coefficients due to the lack of glazing.
    """
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float):
        """
        Initialize an unglazed solar thermal collector.
        
        Parameters
        ----------
        name : str
            Unique identifier for the collector
        tilt : float
            Tilt angle in degrees
        azimuth : float
            Azimuth angle in degrees
        surface_area : float
            Collector aperture area in m²
        latitude : float
            Location latitude in degrees
        longitude : float
            Location longitude in degrees
        """
        efficiency_0 = (0.8 + 0.9) / 2
        efficiency_1 = (15.0 + 25.0) / 2
        super().__init__(name, tilt, azimuth, surface_area, latitude, longitude, efficiency_0 = efficiency_0, efficiency_1 = efficiency_1)

class FlatPlateCollector(SolarCollector):
    """
    Flat-plate solar thermal collector.
    
    Flat-plate collectors are the most common type for residential hot water applications.
    They feature a transparent glazing layer and insulation, offering a good balance between
    efficiency and cost. Typical efficiency: 70-80% zero-loss, 3.5-6.0 W/(m²·K) linear loss.
    """
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float):
        """
        Initialize a flat-plate solar thermal collector.
        
        Parameters
        ----------
        name : str
            Unique identifier for the collector
        tilt : float
            Tilt angle in degrees
        azimuth : float
            Azimuth angle in degrees
        surface_area : float
            Collector aperture area in m²
        latitude : float
            Location latitude in degrees
        longitude : float
            Location longitude in degrees
        """
        efficiency_0 = (0.7 + 0.8) / 2
        efficiency_1 = (3.5 + 6.0) / 2
        super().__init__(name, tilt, azimuth, surface_area, latitude, longitude, efficiency_0 = efficiency_0, efficiency_1 = efficiency_1)

class EvacuatedTubeCollector(SolarCollector):
    """
    Evacuated-tube solar thermal collector.
    
    Evacuated-tube collectors use a vacuum seal to minimize heat losses, making them ideal for
    high-temperature applications and cold climates. They have slightly lower zero-loss efficiency
    (60-75%) but significantly lower heat loss coefficients (1.0-2.0 W/(m²·K)) compared to flat-plate
    collectors, resulting in better performance in adverse conditions.
    """
    def __init__(self, name: str, tilt: float, azimuth: float, surface_area: float, latitude: float, longitude: float):
        """
        Initialize an evacuated-tube solar thermal collector.
        
        Parameters
        ----------
        name : str
            Unique identifier for the collector
        tilt : float
            Tilt angle in degrees
        azimuth : float
            Azimuth angle in degrees
        surface_area : float
            Collector aperture area in m²
        latitude : float
            Location latitude in degrees
        longitude : float
            Location longitude in degrees
        """
        efficiency_0 = (0.6 + 0.75) / 2
        efficiency_1 = (1.0 + 2.0) / 2
        super().__init__(name, tilt, azimuth, surface_area, latitude, longitude, efficiency_0 = efficiency_0, efficiency_1 = efficiency_1)
