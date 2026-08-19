"""
Environmental Data Providers

This module provides abstract and concrete implementations for supplying environmental
data (temperature, irradiation, etc.) to simulations. Multiple backend strategies are
supported: CSV files and external APIs.

Classes
-------
EnvironmentalDataProvider
    Abstract base class defining the interface for environmental data providers.
CSVEnvironmentalProvider
    Reads environmental data from CSV files with automatic resampling.
APIEnvironmentalProvider
    Fetches environmental data from external APIs in real-time.
"""

from abc import ABC, abstractmethod
import pandas as pd
from energy_system_control.core.base_classes import EnvironmentalData
from energy_system_control.sim.config import SimulationConfig
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import resample_with_interpolation, C2K
from energy_system_control.io.weather_api import WeatherAPI
import os


class EnvironmentalDataProvider(ABC):
    """
    Abstract base class for environmental data providers.

    Defines the interface that all environmental data providers must implement.
    Subclasses must provide methods to retrieve environmental data at specific
    simulation timesteps.
    """

    @abstractmethod
    def get_environmental_data(self, time_id: int, current_time: pd.Timestamp) -> EnvironmentalData:
        """
        Retrieve environmental data for the given simulation timestep.

        Parameters
        ----------
        time_id : int
            Simulation timestep index (0-indexed).
        current_time : pd.Timestamp
            Current simulation time.

        Returns
        -------
        EnvironmentalData
            Environmental data object containing temperature, irradiation, and
            other relevant environmental variables.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        pass


class CustomEnvironmentalProvider(EnvironmentalDataProvider):
    """
    Environmental data provider that reads from CSV files.

    Loads environmental data from a CSV file using the specified datetime column,
    automatically resamples data to match the simulation time step, and provides
    environmental data on-demand for each simulation timestep.

    The CSV file should contain a datetime column and columns for each environmental
    variable (e.g., temperature_ambient, direct_irradiation, diffuse_irradiation).

    Examples
    --------
    >>> csv_file = "weather_data.csv"
    >>> provider = CSVEnvironmentalProvider(csv_file)
    >>> cfg = SimulationConfig(time_step_s=300, time_start_h=0, time_end_h=24)
    >>> provider.initialize(cfg)
    >>> env_data = provider.get_environmental_data(time_id=0, current_time=pd.Timestamp('2024-01-01'))

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing environmental data.
    var_types : dict, optional
        Dictionary mapping variable names to their type ('intensive' or 'extensive').
        Intensive variables (e.g., temperature) are interpolated; extensive variables
        (e.g., irradiation) are aggregated during resampling. Default is 'intensive'
        for all variables.

    Attributes
    ----------
    csv_path : str
        Path to the CSV file.
    var_types : dict
        Variable type specifications.
    data : dict
        Loaded and resampled environmental data arrays, keyed by variable name.
    datetime_index : pd.DatetimeIndex, optional
        Datetime index of the loaded data.

    Notes
    -----
    Expected CSV columns:
        - datetime : datetime
            Timestamp for each data point.
        - temperature_ambient : float
            Ambient temperature [K].
        - temperature_cold_water : float, optional
            Cold water temperature [K].
        - direct_irradiation : float
            Direct irradiation [W/m²].
        - diffuse_irradiation : float
            Diffuse irradiation [W/m²].

    Additional custom columns may be included and will be loaded automatically.
    """

    def __init__(self, data_path: str | None = None, filename:str | None = None, column_names: dict[str,str] | None = None, df: pd.DataFrame | None = None, var_types=None):
        """
        Initialize the CSV environmental data provider.

        Parameters
        ----------
        data_path : str, optional
            Path to the data folder in the project
        filename: str, optional
            Name of the CSV file containing the environmental data.
        column_names: dict, optional
            A dictionary mapping the required column names to the ones in the file (format: {'new_name': 'old_name'})
        df: pandas Dataframe, optional
            A pandas DataFrame containing the environmental data. If provided, the `data_path` and `filename` parameters are not used. If not provided, the data is loaded from the csv file
        """
        self.csv_path = os.path.join(data_path, filename) if data_path is not None else None
        self.column_names = column_names
        self.raw_data = df
        self.data = {}
        self.datetime_index = None

    def initialize(self, state: SimulationState | None = None, cfg: SimulationConfig | None = None):
        """
        Load and resample environmental data from CSV file.

        Reads the CSV file, resamples all data columns to match the simulation
        time step, and stores the result in the `data` attribute for fast
        retrieval during simulation.

        Parameters
        ----------
        cfg : SimulationConfig
            Simulation configuration containing time step, start, and end times.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        KeyError
            If required columns are missing from the CSV file.
        """
        if self.raw_data is None:
            self.raw_data = pd.read_csv(self.csv_path, parse_dates=["datetime"])
            self.raw_data = self.raw_data.set_index("datetime")
        if self.column_names is not None:
            self.raw_data.rename(columns = self.column_names)

        target_freq = f"{int(cfg.time_step_h*3600)}s"

        for col in self.raw_data.columns:

            arr = resample_with_interpolation(
                self.raw_data[[col]],
                target_freq=target_freq,
                simulation_end_s=cfg.simulation_end_h*3600.0,
                simulation_start_datetime = cfg.simulation_start_datetime,
                var_type='intensive',
            )

            # flatten (N,1) → (N,)
            self.data[col] = arr.flatten()

    def get_environmental_data(self, time_id: int, current_time: pd.Timestamp) -> EnvironmentalData:
        """
        Get environmental data for a specific simulation timestep.

        Retrieves pre-loaded and resampled environmental data at the specified
        timestep index.

        Parameters
        ----------
        time_id : int
            Simulation timestep index (0-indexed).
        current_time : pd.Timestamp
            Current simulation time (informational, not used for data lookup).

        Returns
        -------
        EnvironmentalData
            Environmental data object containing:
            - temperature_ambient : ambient temperature [K]
            - temperature_cold_water : cold water temperature [K]
            - direct_irradiation : direct irradiation [W/m²]
            - diffuse_irradiation : diffuse irradiation [W/m²]

        Raises
        ------
        IndexError
            If time_id exceeds the available data range.
        """
        temp_env_data = {}
        for key in {'temperature_ambient', 'temperature_cold_water', 'direct_irradiation', 'diffuse_irradiation'}:
            if key in self.data.keys():
                temp_env_data[key] = self.data[key][time_id]
            else:
                temp_env_data[key] = None
            if "temperature" in key:
                if temp_env_data[key] < 200:  # Assuming we never work with temperatures below 200K
                    temp_env_data[key] = C2K(temp_env_data[key])
        return EnvironmentalData(
            temperature_ambient=temp_env_data["temperature_ambient"],
            temperature_cold_water=temp_env_data["temperature_cold_water"],
            direct_irradiation=temp_env_data["direct_irradiation"],
            diffuse_irradiation=temp_env_data["diffuse_irradiation"]
        )


class APIEnvironmentalProvider(EnvironmentalDataProvider):
    """
    Environmental data provider that fetches data from external APIs.

    Retrieves environmental data from external weather or environmental APIs
    in real-time. Useful for live simulations or when historical data is not
    available.

    Examples
    --------
    >>> from some_api_module import WeatherAPIClient
    >>> api = WeatherAPIClient(api_key="your_key")
    >>> provider = APIEnvironmentalProvider(latitude=45.5, longitude=8.9, api_client=api)
    >>> env_data = provider.get_environmental_data(time_id=0, current_time=pd.Timestamp('2024-01-01'))

    Parameters
    ----------
    latitude : float
        Latitude of the location for which to fetch environmental data [degrees].
    longitude : float
        Longitude of the location for which to fetch environmental data [degrees].
    api_client : object
        Client object implementing API calls. Must have a `get_weather()` method
        that accepts lat, lon, and time parameters and returns a dict with
        'temperature', 'dni', and 'dhi' keys.

    Attributes
    ----------
    latitude : float
        Location latitude.
    longitude : float
        Location longitude.
    api_client : object
        API client instance.

    Notes
    -----
    The API client's `get_weather()` method should return a dictionary with:
        - 'temperature' : float
            Ambient temperature [K].
        - 'dni' : float
            Direct normal irradiance [W/m²].
        - 'dhi' : float
            Diffuse horizontal irradiance [W/m²].

    Real-time data fetching may have latency impacts on simulation performance.
    Consider caching results for repeated queries at the same time step.
    """

    def __init__(self, latitude: float, longitude: float, api_client: WeatherAPI):
        """
        Initialize the API environmental data provider.

        Parameters
        ----------
        latitude : float
            Location latitude [degrees].
        longitude : float
            Location longitude [degrees].
        api_client : WeatherAPI
            Configured API client for environmental data retrieval.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.api_client = api_client

    def get_environmental_data(self, time_id: int, current_time: pd.Timestamp) -> EnvironmentalData:
        """
        Fetch environmental data from the API for the current simulation time.

        Queries the external API for environmental data at the specified location
        and time, returning the result as an EnvironmentalData object.

        Parameters
        ----------
        time_id : int
            Simulation timestep index (may be informational for logging).
        current_time : pd.Timestamp
            Current simulation time. Used to query the API for weather at this time.

        Returns
        -------
        EnvironmentalData
            Environmental data object containing:
            - temperature_ambient : ambient temperature [K]
            - direct_irradiation : direct normal irradiance [W/m²]
            - diffuse_irradiation : diffuse horizontal irradiance [W/m²]
            - temperature_cold_water : not populated by API provider (None)

        Raises
        ------
        ConnectionError
            If the API request fails or the connection is unavailable.
        KeyError
            If the API response does not contain required fields.
        ValueError
            If the API client returns invalid data types.

        Notes
        -----
        The API call is made synchronously at each timestep. For performance-critical
        simulations, consider implementing a caching mechanism.
        """

        data = self.api_client.get_current_weather(
            lat=self.latitude,
            lon=self.longitude,
            time=current_time
        )

        return EnvironmentalData(
            temperature_ambient=data["temperature"],
            direct_irradiation=data["dni"],
            diffuse_irradiation=data["dhi"]
        )