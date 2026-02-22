from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from typing import Literal
from energy_system_control.sim.state import SimulationState
from energy_system_control.components.base import Component
AlignMethod = Literal["raise", "ffill", "linear"]

class Predictor(ABC):
    variable_to_predict: str | None

    def __init__(self, variable_to_predict: str):
        self.variable_to_predict = variable_to_predict
    
    @abstractmethod
    def predict(
        self,
        horizon: float,  # In seconds
        state: SimulationState
    ) -> pd.DataFrame:
        """
        Return predictions from (now, now+horizon] on a dt grid.

        Output: DataFrame indexed by target timestamps, columns = variables.
        """
    
 
class OfflineForecastPredictor(Predictor):
    forecast_df: pd.DataFrame          # MultiIndex(issue_time, valid_time), columns=variables
    issue_level: str
    valid_level: str
    align: AlignMethod

    def __init__(self, 
                 forecast_df: pd.DataFrame, 
                 variable_to_predict: str,
                 issue_level: str = "issue_time",
                 valid_level: str = "valid_time",
                 align: AlignMethod = "ffill"
                 ):
        super().__init__(variable_to_predict)
        self.forecast_df = forecast_df
        self.issue_level = issue_level
        self.valid_level = valid_level
        self.align = align


    def _select_issue_time(self, now: pd.Timestamp) -> pd.Timestamp:
        issue_times = self.forecast_df.index.get_level_values(self.issue_level).unique().sort_values()
        eligible = issue_times[issue_times <= now]
        if len(eligible) == 0:
            raise ValueError(f"No forecast available at or before now={now}.")
        return eligible[-1]

    def predict(
        self,
        horizon: float,  # In seconds
        state: SimulationState
    ) -> pd.DataFrame:
        """
        Return predictions from (now, now+horizon] on a dt grid.

        Parameters:
        -----------
        time : float
            Current simulation time, in seconds.
        simulation_start_datetime : pd.Timestamp
            The start datetime of the simulation.
        horizon : float
            The prediction horizon in hours.
        variables : Sequence[str]
            The variables to predict.
        dt : float
            The time step in seconds.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the predictions for the specified variables over the prediction horizon.

        Raises:
        -------
        KeyError
            If any of the specified variables are missing in the forecast DataFrame.
        ValueError
            If the forecast run does not cover the requested horizon with the specified time step.
        """
        
        # Determine best time for prediction
        now = state.simulation_start_datetime + pd.Timedelta(seconds=state.time)
        issue = self._select_issue_time(now)
        # Build target grid: (now, now+horizon] at dt
        start = now + pd.to_timedelta(state.time_step, 'second')
        end = now + pd.to_timedelta(horizon, 'hours')
        target_index = pd.date_range(start=start, end=end, freq=pd.to_timedelta(state.time_step, 'second'))
        # Slice forecast run
        run = self.forecast_df.xs(issue, level=self.issue_level)
        # Checking that the required horizon is available
        first_valid = run.index.min()
        last_valid = run.index.max()
        if start < first_valid or end > last_valid:
            raise ValueError(
                f"Forecast run issued at {issue} covers [{first_valid}, {last_valid}] "
                f"but requested [{start}, {end}]."
    )
        # Resample if needed
        if state.time_step == self.dt_native:
            out = run.reindex(target_index)
        else:
            out = self._align_to_grid(run, target_index, method=self.align)
        # Ensure variables exist
        if self.variable_to_predict not in out.columns:
            raise KeyError(f"Missing variables in forecast_df: {self.variable_to_predict}")

        return out.loc[target_index, self.variable_to_predict]
    
    def _align_to_grid(
        self,
        series_df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        method: str,
    ) -> pd.DataFrame:

        """
        Aligns a time series DataFrame to a target datetime index using specified alignment method.

        Parameters:
        -----------
        series_df : pd.DataFrame
            The input time series DataFrame to be aligned. Must have a monotonic datetime index.
        target_index : pd.DatetimeIndex
            The target datetime index to which the series will be aligned.
        method : str
            The alignment method to use. Supported methods are:
            - "ffill": Forward fill (zero-order hold) to align the series.
            - "linear": Linear interpolation to align the series.

        Returns:
        --------
        pd.DataFrame
            The aligned DataFrame with the target datetime index.

        Raises:
        -------
        ValueError
            If the specified alignment method is not recognized.
        """

        # Ensure monotonic index for time-based operations
        series_df = series_df.sort_index()

        if method == "ffill":   # zero-order hold
            # First reindex to union so ffill has anchors, then select target
            union = target_index.union(series_df.index)
            return series_df.reindex(union).ffill().reindex(target_index)

        if method == "linear":
            union = target_index.union(series_df.index)
            return series_df.reindex(union).interpolate(method="time").reindex(target_index)

        raise ValueError(f"Unknown alignment method: {method}")


    @staticmethod
    def build_forecast_df(daily_frames: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Each daily frame must contain:
        - a column 'issue_time' (same for all rows in that frame)
        - a column 'valid_time'
        - variable columns e.g. DHI/DNI
        """
        df = pd.concat(daily_frames, ignore_index=True)
        df["issue_time"] = pd.to_datetime(df["issue_time"])
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.set_index(["issue_time", "valid_time"]).sort_index()
        return df


class DailyProfilePredictor(Predictor):
    profile: pd.DataFrame
    original_frequency: float
    """
    This class implements a generic predictor that does the 
    prediction based on a fixed daily profile. Very simple, but kind of useful
    especially for testing
    """
    
    def __init__(self, variable_to_predict: str, profile: pd.DataFrame):
        self.check_raw_profile(profile)
        self.profile = profile
        self.profile.index = self.profile.index * 3600  # Converting profile indeces to seconds
        self.original_frequency = self.profile.index.to_series().diff().median()
        super().__init__(variable_to_predict=variable_to_predict)
        
    def predict(self, horizon, state):
        """
        Return predictions from (now, now+horizon] on a dt grid.

        Parameters:
        -----------
        horizon : float
            The prediction horizon in hours.
        state : SimulationState
            The current state of the simulation.

        Returns:
        --------
        np.array
            A Numpy array containing the predictions for the specified variable over the prediction horizon.
        """
        # Determine the current time
        if state.simulation_start_datetime is None:
            raise(ValueError("Simulation start datetime not set in state."))
        now = state.simulation_start_datetime + pd.Timedelta(seconds=state.time)

        # Build target grid: (now, now+horizon] at dt
        start = now  #  + pd.to_timedelta(state.time_step, 'second')
        end = now + pd.to_timedelta(horizon, 'hours')
        target_index = pd.date_range(start=start, end=end, freq=pd.to_timedelta(state.time_step, 'second'))

        # Repeat the daily profile to cover the prediction horizon
        profile_start = start.normalize()
        profile_repeated = pd.concat([self.profile] * (int(horizon * 3600 / (state.time_step * (self.profile.index[-1] - self.profile.index[0]))) + 1))

        # Align the repeated profile to the target index
        profile_repeated.index = pd.date_range(start=profile_start, periods=len(profile_repeated), freq=pd.to_timedelta(self.original_frequency, 'second'))
        profile_repeated = profile_repeated.reindex(target_index, method='ffill')

        return profile_repeated[self.variable_to_predict].values[:int(horizon * 3600 / state.time_step)]

    def check_raw_profile(self, profile):
        assert profile.index.min() == 0.0
        assert profile.index.max() <= 24.0
        assert profile.index.max() >= 23.0

class PerfectTimeSeriesPredictor(Predictor):
    read_component: str
    def __init__(self, read_component: str, variable_to_predict: str = None):
        super().__init__(variable_to_predict=variable_to_predict)
        self.read_component = read_component

    def initialize(self, environment):
        self.data = environment.components[self.read_component].data

    def predict(self, horizon, state):
        return self.data[state.time_id: np.where(state.time_vector_for_prediction == state.time + horizon*3600)[0][0]] / state.time_step