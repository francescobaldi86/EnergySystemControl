import pandas as pd
import numpy as np
import pvlib
from typing import List, Dict, Any, Literal
from datetime import datetime

TimeAlignment = Literal["datetime", "yearly", "daily"]
TimeMatch = Literal["nearest", "forward", "exact"]
DemandType = Literal['electricity', 'fluid']
VariableType = Literal['energy', 'power', 'volume', 'mass', 'temperature']
VariableUnit = Literal['Wh', 'kWh', 'MWh', 'W', 'kW', 'MW', 'l', 'm3', 'kg', 'C', 'K']

def resample_with_interpolation(
    df: pd.DataFrame,
    target_freq: str,
    simulation_end_s: float | None = None,
    var_type: Literal["extensive", "intensive"] = "extensive",
    simulation_start_datetime: datetime | None = None,
    time_alignment: TimeAlignment = "datetime",
    match_method: TimeMatch = "nearest",
    tolerance: pd.Timedelta = pd.Timedelta(minutes=30)
) -> np.ndarray:
    """
    Resample a time series DataFrame to a new frequency, handling both
    upsampling (with interpolation) and downsampling (with aggregation).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex.

    target_freq : str
        New frequency (e.g. "15min", "1h", "1D").

    simulation_end_s : float, optional
        Simulation duration in seconds. If provided and longer than the
        input data, the input time series is repeated as necessary.

    var_type : {"extensive", "intensive"}, optional
        Type of variable.

        - "extensive": values are summed when downsampling and scaled
          proportionally when upsampling.
        - "intensive": values are averaged when downsampling and
          interpolated when upsampling.

    simulation_start_datetime : datetime, optional
        Start datetime of the simulation. If None, no time alignment
        is performed.

    time_alignment : {"datetime", "yearly", "daily"}, optional
        Defines how the input data should be aligned to the simulation:

        - "datetime": match the complete datetime.
        - "yearly": match month, day and time, ignoring the year.
        - "daily": match only the time of day.

    Returns
    -------
    np.ndarray
        Resampled time series as a one-dimensional NumPy array.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    if simulation_start_datetime is not None:
        matching_timestamp = _find_simulation_start_matching_index(
                index=df.index,
                simulation_start=pd.Timestamp(simulation_start_datetime),
                time_alignment=time_alignment,
                tolerance=tolerance,
                match_method=match_method
            )
        df = df.loc[matching_timestamp:].copy()
    else:
        df = df.copy()

    # ------------------------------------------------------------------
    # Repeat the input data if necessary
    # ------------------------------------------------------------------

    if simulation_end_s is not None:
        original_step = df.index[1] - df.index[0]
        period = df.index[-1] - df.index[0] + original_step

        n_repeat = int(
            np.ceil(
                pd.to_timedelta(simulation_end_s, unit="s") / period
            )
        )

        dfs = []

        for i in range(n_repeat):
            df_copy = df.copy()
            df_copy.index = df_copy.index + i * period
            dfs.append(df_copy)

        df = pd.concat(dfs)

        # Trim to exact simulation end
        simulation_end = (
            df.index[0] + pd.to_timedelta(simulation_end_s, unit="s")
        )
        df = df[df.index <= simulation_end]

    # ------------------------------------------------------------------
    # Determine original resolution
    # ------------------------------------------------------------------

    original_freq = pd.infer_freq(df.index)

    if original_freq is None:
        original_step = df.index.to_series().diff().median()
    else:
        original_step = pd.to_timedelta(
            pd.tseries.frequencies.to_offset(original_freq)
        )

    target_step = pd.to_timedelta(
        pd.tseries.frequencies.to_offset(target_freq)
    )

    # ------------------------------------------------------------------
    # Add one final value to avoid losing the last interval
    # ------------------------------------------------------------------

    last_index = df.index[-1] + original_step
    df.loc[last_index] = df.iloc[-1]

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    if target_step >= original_step:
        # Downsampling

        match var_type:
            case "extensive":
                output = df.resample(target_freq).sum()

            case "intensive":
                output = df.resample(target_freq).mean()

            case _:
                raise ValueError(
                    f"Invalid var_type: {var_type!r}. "
                    "Expected 'extensive' or 'intensive'."
                )

    else:
        # Upsampling

        match var_type:
            case "extensive":
                output = df.resample(target_freq).ffill()
                output = output * (target_step / original_step)

            case "intensive":
                new_index = pd.date_range(
                    start=df.index[0],
                    end=df.index[-1],
                    freq=target_freq,
                    tz=df.index.tz,
                )

                output = (
                    df.reindex(new_index)
                    .interpolate(method="time")
                )

            case _:
                raise ValueError(
                    f"Invalid var_type: {var_type!r}. "
                    "Expected 'extensive' or 'intensive'."
                )

    return output.to_numpy().ravel()


def _find_simulation_start_matching_index(
    index: pd.DatetimeIndex,
    simulation_start: pd.Timestamp,
    time_alignment: TimeAlignment,
    tolerance: pd.Timedelta | None = pd.Timedelta(minutes=30),
    match_method: TimeMatch = "nearest"
) -> pd.Timestamp:

    match time_alignment:

        case "datetime":
            target = simulation_start

            # target_index = pd.DatetimeIndex([target])
            target_index = index

        case "yearly":
            # Ignore the source year.
            #
            # Convert every source timestamp to the simulation year,
            # preserving month, day and time.
            try:
                target_index = pd.DatetimeIndex(
                    [
                        timestamp.replace(year=simulation_start.year)
                        for timestamp in index
                    ]
                )
            except ValueError as exc:
                raise ValueError(
                    "Cannot map the source timestamps to the "
                    f"simulation year {simulation_start.year}. "
                    "This may be caused by February 29 in a "
                    "non-leap simulation year."
                ) from exc

            target = simulation_start

        case "daily":
            # Replace the date of every source timestamp with the
            # simulation date, preserving the time of day.
            target_index = pd.DatetimeIndex(
                [
                    timestamp.replace(
                        year=simulation_start.year,
                        month=simulation_start.month,
                        day=simulation_start.day,
                    )
                    for timestamp in index
                ]
            )

            target = simulation_start

        case _:
            raise ValueError(
                f"Invalid time_alignment: {time_alignment!r}."
            )

    positions = target_index.get_indexer(
        [target],
        method=match_method,
        tolerance=tolerance,
    )

    position = positions[0]

    if position == -1:
        raise ValueError(
            f"Could not find a matching timestamp for "
            f"{simulation_start} using "
            f"time_alignment={time_alignment!r} "
            f"within tolerance={tolerance}."
        )

    return index[position]


def C2K(T):
    """
    Convert temperature from Celsius to Kelvin.
    
    Parameters
    ----------
    T : float, int, list, or np.ndarray
        Temperature value(s) in Celsius.
    
    Returns
    -------
    float, int, list, or np.ndarray
        Temperature in Kelvin, preserving the input type.
        
    Examples
    --------
    >>> C2K(0)
    273.15
    >>> C2K([0, 25, 100])
    [273.15, 298.15, 373.15]
    >>> C2K(np.array([0, 25]))
    array([273.15, 298.15])
    """
    if isinstance(T, list):
        result = np.asarray(T) + 273.15
        return result.tolist()
    else:
        return T + 273.15


def K2C(T):
    """
    Convert temperature from Kelvin to Celsius.
    
    Parameters
    ----------
    T : float, int, list, or np.ndarray
        Temperature value(s) in Kelvin.
    
    Returns
    -------
    float, int, list, or np.ndarray
        Temperature in Celsius, preserving the input type.
        
    Examples
    --------
    >>> K2C(273.15)
    0.0
    >>> K2C([273.15, 298.15, 373.15])
    [0.0, 25.0, 100.0]
    >>> K2C(np.array([273.15, 298.15]))
    array([0., 25.])
    """
    if isinstance(T, list):
        result = np.asarray(T) - 273.15
        return result.tolist()
    else:
        return T - 273.15

def find_object_of_type(object_type: Any, object_pool: List[Any] | Dict[str, Any]) -> Any:
    """
    Finds and returns the first component of the specified type in the list of components.

    Args:
        object_type (Object): The type of component to search for.
        object_pool (List[Object] or Dict[str, Object]): The list (or dict) of objects to search through.

    Returns:
        Object: The first object found of the specified type. Returns None if nothing is found

    """
    if isinstance(object_pool, dict):
        object_pool = [v for k, v in object_pool.items()]
    for object in object_pool:
        if isinstance(object, object_type):
            return object
    return None

def check_datetime_index(df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        

def calculate_solar_angles(latitude: float, longitude: float, timestamps: pd.DatetimeIndex):
    """
    Calculate solar zenith and azimuth for a location and timestamps.

    Parameters
    ----------
    latitude : float
        Latitude in degrees (positive north)
    longitude : float
        Longitude in degrees (positive east)
    timestamps : pd.DatetimeIndex
        Times for which to compute solar positions (timezone-aware is preferred)

    Returns
    -------
    zenith : pd.Series
        Solar zenith angle in degrees
    azimuth : pd.Series
        Solar azimuth angle in degrees
    """
    # Assume UTC if no timezone
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize('UTC')
    
    solpos = pvlib.solarposition.get_solarposition(
        time=timestamps,
        latitude=latitude,
        longitude=longitude
    )

    zenith = solpos['zenith']
    azimuth = solpos['azimuth']

    return zenith, azimuth

def calculate_effective_irradiance(solar_zenith: float, solar_azimuth: float, surface_tilt: float, surface_azimuth: float, direct_irradiation: float, diffuse_irradiation: float):
    """
    Calculate the effective irradiance incident on a tilted surface.

    The direct component is projected onto the surface using the angle of
    incidence. Negative projections, which occur when the sun is behind the
    surface, are clipped to zero. The diffuse component is estimated as
    isotropic sky diffuse irradiance over the visible hemisphere.

    Parameters
    ----------
    solar_zenith : float
        Solar zenith angle in radians.
    solar_azimuth : float
        Solar azimuth angle in radians.
    surface_tilt : float
        Surface tilt from horizontal in radians.
    surface_azimuth : float
        Surface azimuth angle in radians.
    direct_irradiation : float
        Direct normal irradiance incident on the surface, in W/m2.
    diffuse_irradiation : float
        Diffuse horizontal irradiance, in W/m2

    Returns
    -------
    float
        Effective plane-of-array irradiance in the same units as the input
        irradiances.
    """
    cos_theta = (
                np.cos(solar_zenith) * np.cos(surface_tilt) +
                np.sin(solar_zenith) * np.sin(surface_tilt) * np.cos(solar_azimuth - surface_azimuth)
            )
    cos_theta = max(cos_theta, 0)

    # POA irradiance
    return direct_irradiation * cos_theta + diffuse_irradiation * (1 + np.cos(surface_tilt)) / 2
    


class NodeImbalanceError(Exception):
    pass

class StorageError(Exception):
    pass

class OnOffComponentError(BaseException):
    pass
