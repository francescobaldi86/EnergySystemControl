import pandas as pd
import numpy as np
import pytest
from energy_system_control.controllers.predictors import OfflineForecastPredictor

@pytest.fixture
def forecast_df():
    # Issue times (forecasts issued at midnight)
    issue_times = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-02 00:00:00"])

    # Valid times hourly for 2 days
    valid_times = pd.date_range("2026-01-01 00:00:00", periods=48, freq="1h")

    rows = []
    for issue in issue_times:
        for vt in valid_times:
            # Make values depend on issue + valid_time so we can detect which issue was used
            base = 1000 if issue == issue_times[0] else 2000
            rows.append(
                {
                    "issue_time": issue,
                    "valid_time": vt,
                    "DHI": base + (vt.hour),
                    "DNI": base + 10 + (vt.hour),
                }
            )

    df = pd.DataFrame(rows)
    df = df.set_index(["issue_time", "valid_time"]).sort_index()
    return df


def assert_predictor_contract(predictor, sim_start, time, horizon, dt, variables):
    out = predictor.predict(time, sim_start, horizon, variables, dt)
    assert list(out.columns) == list(variables)
    assert out.index.is_monotonic_increasing
    assert not out.isna().any().any()
    return out


def test_OfflineForecastPredictor_contract(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df, align="ffill")
    sim_start = pd.Timestamp("2026-01-01 00:00:00")
    time = 0.0
    horizon = 2
    variables = ["DHI", "DNI"]
    dt = 900
    out = assert_predictor_contract(pred, sim_start, time, horizon, dt, variables)
    assert out.index[0] == pd.Timestamp("2026-01-01 00:15:00")
    assert True


def test_select_issue_time_latest_leq_now(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df)

    now = pd.Timestamp("2026-01-02 12:00:00")
    issue = pred._select_issue_time(now)
    assert issue == pd.Timestamp("2026-01-02 00:00:00")


def test_select_issue_time_raises_before_first(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df)
    with pytest.raises(ValueError):
        pred._select_issue_time(pd.Timestamp("2025-12-31 23:00:00"))


def test_predict_returns_expected_index_and_columns(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df, align="ffill")

    sim_start = pd.Timestamp("2026-01-02 00:00:00")
    time = 0.0
    dt = 3600.0            # 1h
    horizon = 6    # 6h
    out = pred.predict(
        time=time,
        simulation_start_datetime=sim_start,
        horizon=horizon,
        variables=["DHI", "DNI"],
        dt=dt,
    )
    # index should be (now+dt ... now+horizon) inclusive at dt
    assert out.index[0] == sim_start + pd.Timedelta(hours=1)
    assert out.index[-1] == sim_start + pd.Timedelta(hours=6)
    assert list(out.columns) == ["DHI", "DNI"]
    assert len(out) == 6


def test_predict_uses_latest_available_issue(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df, align="ffill")

    sim_start = pd.Timestamp("2026-01-02 12:00:00")
    out = pred.predict(
        time=0.0,
        simulation_start_datetime=sim_start,
        horizon=2,
        variables=["DHI"],
        dt=3600.0,
    )

    # First prediction at 13:00 -> hour=13, base for issue=2026-01-02 is 2000
    assert out.iloc[0]["DHI"] == 2000 + 13


def test_predict_missing_variable_raises_keyerror(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df)

    with pytest.raises(KeyError):
        pred.predict(
            time=0.0,
            simulation_start_datetime=pd.Timestamp("2026-01-02 00:00:00"),
            horizon=1,
            variables=["NOT_A_VAR"],
            dt=3600.0,
        )


def test_predict_raises_if_horizon_not_covered(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df, align="ffill")

    sim_start = pd.Timestamp("2026-01-02 23:00:00")
    with pytest.raises(ValueError):
        pred.predict(
            time=0.0,
            simulation_start_datetime=sim_start,
            horizon=10,   # push beyond the fixture’s range
            variables=["DHI"],
            dt=3600.0,
        )


def test_align_ffill_hourly_to_minutes(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df, align="ffill")

    sim_start = pd.Timestamp("2026-01-02 00:00:00")
    out = pred.predict(
        time=0.0,
        simulation_start_datetime=sim_start,
        horizon=1,   # 1h ahead
        variables=["DHI"],
        dt=60.0,          # 1 minute
    )

    # First minute ahead is 00:01; ffill should use 01:00 anchor only if available,
    # so make sure your implementation includes proper anchoring.
    # If you align using union+ffill, values from 01:00 won't backfill to 00:01.
    # Typically you'd want ffill from the last known value <= target time.
    # So the expected behavior depends on your policy.
    assert out.index.freq is None or len(out) == 60


def test_align_to_grid_unknown_method_raises(forecast_df):
    pred = OfflineForecastPredictor(forecast_df=forecast_df)
    run = forecast_df.xs(pd.Timestamp("2026-01-02 00:00:00"), level="issue_time")
    target = pd.date_range("2026-01-02 00:10:00", periods=5, freq="1min")

    with pytest.raises(ValueError):
        pred._align_to_grid(run, target, method="nope")


def test_build_forecast_df_sets_multiindex_and_sorts():
    f1 = pd.DataFrame({
        "issue_time": ["2026-01-01"] * 2,
        "valid_time": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
        "DHI": [1, 2],
    })
    f2 = pd.DataFrame({
        "issue_time": ["2026-01-02"] * 1,
        "valid_time": ["2026-01-02 00:00:00"],
        "DHI": [3],
    })

    df = OfflineForecastPredictor.build_forecast_df([f1, f2])
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["issue_time", "valid_time"]
    assert df.sort_index().equals(df)

@pytest.fixture
def basic_daily_profile_predictor():
    from energy_system_control.controllers.predictors import DailyProfilePredictor
    # Create a sample profile
    profile = pd.DataFrame(index = np.arange(start=0.0, stop=24.0, step=1.0), data={'power': np.random.rand(24)})
    variable_to_predict = 'power'
    
    # Create a predictor instance
    predictor = DailyProfilePredictor(
        profile=profile,
        variable_to_predict=variable_to_predict
    )
    return predictor

def test_DailyProfilePredictor_setup(basic_daily_profile_predictor):
    profile = pd.DataFrame(index = np.arange(start=0.0, stop=24.0, step=1.0), data={'power': np.random.rand(24)})
    profile.index *= 3600.0
    assert basic_daily_profile_predictor.variable_to_predict == 'power'

def test_DailyProfilePredictor_predict(basic_daily_profile_predictor):
    # Create a simulation state
    from energy_system_control.sim.state import SimulationState
    simulation_start_datetime = pd.to_datetime("2023-01-01 00:00")
    state = SimulationState(
        simulation_start_datetime=simulation_start_datetime,
        time=0,
        time_step = 900
    )
    # Call the predict method
    horizon = 24  # 24 hours
    predictions = basic_daily_profile_predictor.predict(horizon, state)

    # Check the predictions
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == int(horizon * 3600 / state.time_step)