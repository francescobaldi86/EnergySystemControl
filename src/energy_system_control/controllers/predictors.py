from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from typing import Literal
from collections import deque
from sklearn.neural_network import MLPRegressor
from energy_system_control.sim.state import SimulationState
from energy_system_control.components.base import Component
AlignMethod = Literal["raise", "ffill", "linear"]

class Predictor(ABC):
    variable_to_predict: str | None
    name : str

    def __init__(self, name: str, variable_to_predict: str):
        self.name = name
        self.variable_to_predict = variable_to_predict

    def initialize(self):
        return None
    
    def update(self):
        return None
    
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
    """
    A predictor class that uses pre-computed forecast data to make predictions.

    This class inherits from the Predictor base class and implements a predictor
    that uses pre-computed forecast data stored in a DataFrame. The predictor
    selects the most recent forecast available at the current time and uses it
    to make predictions for the specified horizon.

    Parameters
    ----------
    name : str
        Name of the predictor.
    forecast_df : pd.DataFrame
        A DataFrame containing the forecast data. The DataFrame should have a
        MultiIndex with levels for issue_time and valid_time, and columns for
        the variables to predict.
    variable_to_predict : str
        The name of the variable to predict.
    issue_level : str, optional
        The name of the level in the MultiIndex that contains the issue times.
        Defaults to "issue_time".
    valid_level : str, optional
        The name of the level in the MultiIndex that contains the valid times.
        Defaults to "valid_time".
    align : AlignMethod, optional
        The alignment method to use when aligning the forecast data to the target
        grid. Supported methods are "ffill" (forward fill) and "linear" (linear
        interpolation). Defaults to "ffill".

    Attributes
    ----------
    forecast_df : pd.DataFrame
        The DataFrame containing the forecast data.
    issue_level : str
        The name of the level in the MultiIndex that contains the issue times.
    valid_level : str
        The name of the level in the MultiIndex that contains the valid times.
    align : AlignMethod
        The alignment method to use when aligning the forecast data to the target
        grid.
    """
    forecast_df: pd.DataFrame          # MultiIndex(issue_time, valid_time), columns=variables
    issue_level: str
    valid_level: str
    align: AlignMethod
    dt_native_s: float

    def __init__(self,
                 name: str, 
                 forecast_df: pd.DataFrame, 
                 variable_to_predict: str,
                 issue_level: str = "issue_time",
                 valid_level: str = "valid_time",
                 align: AlignMethod = "ffill"
                 ):
        super().__init__(name = name, variable_to_predict = variable_to_predict)
        self.forecast_df = forecast_df
        self.issue_level = issue_level
        self.valid_level = valid_level
        self.align = align
        self.dt_native_s = self.forecast_df.index.get_level_values(valid_level).to_series().diff().median().seconds


    def _select_issue_time(self, now: pd.Timestamp) -> pd.Timestamp:
        """
        Select the most recent forecast issue time that is available at or before the current time.

        Parameters
        ----------
        now : pd.Timestamp
            The current time.

        Returns
        -------
        pd.Timestamp
            The most recent forecast issue time that is available at or before the current time.

        Raises
        ------
        ValueError
            If no forecast is available at or before the current time.
        """
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
        horizon : float
            The prediction horizon in hours.
        state : SimulationState
            The current state of the simulation (instance of the SimulationState class)

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
        if state.time_step == self.dt_native_s:
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
    
    def __init__(self, name: str, variable_to_predict: str, profile: pd.DataFrame):
        self.check_raw_profile(profile)
        self.profile = profile
        self.profile.index = self.profile.index * 3600  # Converting profile indeces to seconds
        self.original_frequency = self.profile.index.to_series().diff().median()
        super().__init__(name = name, variable_to_predict=variable_to_predict)
        
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
    def __init__(self, name: str, read_component: str, variable_to_predict: str = None):
        super().__init__(name = name, variable_to_predict=variable_to_predict)
        self.read_component = read_component

    def initialize(self, ctx):
        self.data = ctx.environment.components[self.read_component].ts.data

    def predict(self, horizon, state):
        return self.data[state.time_id: np.where(state.time_vector_for_prediction == state.time + horizon*3600)[0][0]]
    

class ANNBasedPredictor(Predictor):
    """
    A predictor class that uses an Artificial Neural Network (ANN) to make predictions.

    This class inherits from the Predictor base class and implements an ANN-based
    prediction model. The predictor uses historical sensor data to train the ANN
    and make predictions about future values, using past data plus the hour of the day 
    as exogenous variable.

    Parameters
    ----------
    prediction_horizon_h : float
        The amount of time into the future to predict, in [h].
    sensor_name : str
        The name of the sensor whose values are being predicted.
    window_size_h : float, optional
        The length of past time steps to use as input features. Defaults to 144.
    retrain_interval_h : float, optional
        The interval of time that basses between model retraining. Defaults to 24.
    min_sample_size_h : float, optional
        The minimum length of time required to train the model. Defaults to 24.
    **ann_kwargs
        Additional keyword arguments to pass to the MLPRegressor.

    Attributes
    ----------
    buffer : deque
        A buffer to store recent sensor values.
    time_buffer : deque
        A buffer to store timestamps corresponding to sensor values.
    step_counter : int
        A counter to keep track of the number of steps.
    model : MLPRegressor
        The ANN model used for prediction.
    is_trained : bool
        A flag indicating whether the model has been trained.
    """
    def __init__(
        self,
        prediction_horizon_h,
        sensor_name,
        name=None,
        window_size_h=24,
        retrain_interval_h=50,
        min_sample_size_h=200,
        **ann_kwargs
    ):
        """
        Initialize the ANNBasedPredictor.

        Parameters
        ----------
        prediction_horizon_h : float
            The number of hours into the future to predict.
        sensor_name : str
            The name of the sensor whose values are being predicted.
        name : str, optional
            The name of this predictor. Defaults to sensor_name if not provided.
        window_size_h : float, optional
            The length of past time steps to use as input features. Defaults to 24.
        retrain_interval_h : float, optional
            The interval of time between model retraining. Defaults to 50.
        min_sample_size_h : float, optional
            The minimum length of time required to train the model. Defaults to 200.
        **ann_kwargs
            Additional keyword arguments to pass to the MLPRegressor.
        """
        # If no name provided, use sensor_name
        if name is None:
            name = sensor_name
            
        # Initialize the base Predictor class
        super().__init__(name=name, variable_to_predict=sensor_name)
        
        self.prediction_horizon_h = prediction_horizon_h
        self.prediction_horizon = None
        self.sensor_name = sensor_name
        self.sensor = None
        self.window_size_h = window_size_h
        self.window_size = None
        self.retrain_interval_h = retrain_interval_h
        self.retrain_interval = None
        self.min_sample_size_h = min_sample_size_h
        self.min_sample_size = None

        self.buffer = deque(maxlen=5000)
        self.time_buffer = deque(maxlen=5000)

        self.step_counter = 0
        self.model = MLPRegressor(**ann_kwargs)
        self.is_trained = False

        # Check that sample sizes are consistent
        if self.min_sample_size_h < self.window_size_h + self.prediction_horizon_h:
            raise ValueError(f'The minimum sample size should be larger than the sum of the window size plus the prediction horizon.\n Values provided are:\n - min_sample_size: {self.min_sample_size_h} h\n - window_size: {self.window_size_h} h\n - prediction_horizon: {self.prediction_horizon_h} h')

    def initialize(self, ctx):
        self.window_size = self.window_size_h * 3600 // ctx.state.time_step
        self.retrain_interval = self.retrain_interval_h * 3600 // ctx.state.time_step
        self.min_sample_size = self.min_sample_size_h * 3600 // ctx.state.time_step
        self.prediction_horizon = self.prediction_horizon_h * 3600 //ctx.state.time_step
        self.sensor = ctx.environment.sensors[self.sensor_name]
    
    def update(self, time_s):
        """
        Update the predictor with a new sensor value and timestamp.

        Parameters
        ----------
        time_s : float
            The simulation time corresponding to the sensor value [s].
        """
        self.buffer.append(self.sensor.get_measurement())
        self.time_buffer.append(time_s)
        # If there are enough samples for training, and it's time to retrain, train the model
        if self.time_buffer[-1] / 3600 >= self.min_sample_size_h:
            if self.time_buffer[-1] / 3600 % self.retrain_interval_h == 0:
                self._train()

    def _train(self):
        X, Y = self._prepare_data()
        if len(X) == 0:
            return

        self.model.fit(X, Y)
        self.is_trained = True
    
    def _prepare_data(self):
        """
        Prepare the data for training the ANN model.

        Returns
        -------
        tuple
            A tuple containing the input features (X) and target values (Y) for training.
        """
        values = np.array(self.buffer)
        times = list(self.time_buffer)
        X = []
        Y = []
        for i in range(len(values) - self.window_size - self.prediction_horizon):
            # lagged values
            lag_part = values[i:i+self.window_size]
            # hour at prediction start
            pred_time = times[i+self.window_size]
            sin_h, cos_h = self._encode_hour(pred_time)
            features = np.concatenate([lag_part, [sin_h, cos_h]])
            target = values[
                i+self.window_size:
                i+self.window_size+self.prediction_horizon
            ]
            X.append(features)
            Y.append(target)
        return np.array(X), np.array(Y)
    
    def predict(self, horizon_h: float, state: SimulationState):
        """
        Make a prediction using the trained ANN model.

        Parameters
        ----------
        current_time : pandas.Timestamp or datetime
            The current timestamp.

        Returns
        -------
        numpy.ndarray or None
            The predicted values for the prediction horizon, or None if not enough data is available.
        """
        # check if the prediction horizon is the same as the internal one
        if horizon_h != self.prediction_horizon_h:
            raise ValueError(f'The required prediction horizon of {horizon_h} h is different from the internal prediction horizon of {self.prediction_horizon} h. Please check')
        # Update predictor model
        self.update(state.time)
        # Provide base prediction if it's too early to have a real one
        if not self.is_trained:
            if len(self.buffer) == 0:
                return np.zeros(self.prediction_horizon)
            else:
                last = self.buffer[-1]
                return np.full(self.prediction_horizon, last)
        # If model is trained, let's go baby!
        latest_window = np.array(self.buffer)[-self.window_size:]
        sin_h, cos_h = self._encode_hour(state.time)
        features = np.concatenate([latest_window, [sin_h, cos_h]])
        return self.model.predict(features.reshape(1, -1))[0]
    
    def _encode_hour(self, time):
        hour = (time/3600) % 24
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)
        return sin_hour, cos_hour
