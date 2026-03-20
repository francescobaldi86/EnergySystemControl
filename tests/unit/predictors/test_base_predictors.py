import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass
from energy_system_control.controllers.predictors import OfflineForecastPredictor, DailyProfilePredictor
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext, Sensor
import os


class TestOfflineForecastPredictor:

    def test_OfflineForecastPredictor_predict(self, simulation_state, forecast_df):
        predictor = OfflineForecastPredictor(
            name="test_predictor",
            forecast_df=forecast_df,
            variable_to_predict="DHI",
            align="ffill"
        )
        horizon = 2  # 2 hours
        predictions = predictor.predict(horizon, simulation_state)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == int(horizon * 3600 / simulation_state.time_step)

    def test_OfflineForecastPredictor_select_issue_time(self, forecast_df):
        predictor = OfflineForecastPredictor(
            name="test_predictor",
            forecast_df=forecast_df,
            variable_to_predict="DHI"
        )
        now = pd.Timestamp("2026-01-02 12:00:00")
        issue_time = predictor._select_issue_time(now)
        assert issue_time == pd.Timestamp("2026-01-02 00:00:00")

    def test_OfflineForecastPredictor_missing_variable(self, forecast_df, simulation_state):
        predictor = OfflineForecastPredictor(
            name="test_predictor",
            forecast_df=forecast_df,
            variable_to_predict="MISSING_VAR"
        )
        with pytest.raises(KeyError):
            predictor.predict(2, simulation_state)


class TestDailyProfilePredictor:

    def test_DailyProfilePredictor_predict(self):
        # Create a sample profile
        profile = pd.DataFrame(
            index=np.arange(start=0.0, stop=24.0, step=1.0),
            data={'power': np.random.rand(24)}
        )

        predictor = DailyProfilePredictor(
            name="test_predictor",
            variable_to_predict="power",
            profile=profile
        )

        state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2023-01-01 00:00:00"),
            time=0,
            time_step=900  # 15 minutes
        )

        horizon = 24  # 24 hours
        predictions = predictor.predict(horizon, state)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == int(horizon * 3600 / state.time_step)


    def test_DailyProfilePredictor_invalid_profile(self):
        # Create an invalid profile (doesn't cover full day)
        invalid_profile = pd.DataFrame(
            index=np.arange(start=0.0, stop=12.0, step=1.0),
            data={'power': np.random.rand(12)}
        )

        with pytest.raises(AssertionError):
            DailyProfilePredictor(
                name="test_predictor",
                variable_to_predict="power",
                profile=invalid_profile
            )
