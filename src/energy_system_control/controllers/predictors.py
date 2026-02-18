from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from typing import Literal
AlignMethod = Literal["raise", "ffill", "linear"]

class Predictor(ABC):
    @abstractmethod
    def predict(
        self,
        time: float,  # Current simulation time
        simulation_start_datetime: pd.Timestamp,
        horizon: float,  # In seconds
        variables: Sequence[str],
        dt: float,  # in seconds,
    ) -> pd.DataFrame:
        """
        Return predictions from (now, now+horizon] on a dt grid.

        Output: DataFrame indexed by target timestamps, columns = variables.
        """
 
@dataclass
class OfflineForecastPredictor(Predictor):
    forecast_df: pd.DataFrame          # MultiIndex(issue_time, valid_time), columns=variables
    issue_level: str = "issue_time"
    valid_level: str = "valid_time"
    align: AlignMethod = "ffill"
    dt_native: pd.Timedelta = pd.Timedelta(hours=1)

    def _select_issue_time(self, now: pd.Timestamp) -> pd.Timestamp:
        issue_times = self.forecast_df.index.get_level_values(self.issue_level).unique().sort_values()
        eligible = issue_times[issue_times <= now]
        if len(eligible) == 0:
            raise ValueError(f"No forecast available at or before now={now}.")
        return eligible[-1]

    def predict(
        self,
        time: float,  # Current simulation time
        simulation_start_datetime: pd.Timestamp,
        horizon: float,  # In hours
        variables: Sequence[str],
        dt: float,  # in seconds,
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
        now = simulation_start_datetime + pd.Timedelta(seconds=time)
        issue = self._select_issue_time(now)
        # Build target grid: (now, now+horizon] at dt
        start = now + pd.to_timedelta(dt, 'second')
        end = now + pd.to_timedelta(horizon, 'hours')
        target_index = pd.date_range(start=start, end=end, freq=pd.to_timedelta(dt, 'second'))
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
        if dt == self.dt_native:
            out = run.reindex(target_index)
        else:
            out = self._align_to_grid(run, target_index, method=self.align)
        # Ensure variables exist
        missing = [v for v in variables if v not in out.columns]
        if missing:
            raise KeyError(f"Missing variables in forecast_df: {missing}")

        return out.loc[target_index, list(variables)]
    
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