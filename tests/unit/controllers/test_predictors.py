# test_predictors.py
import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass
from energy_system_control.controllers.predictors import OfflineForecastPredictor, DailyProfilePredictor, ANNBasedPredictor
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext, Sensor

class MockSensor(Sensor):
    def __init__(self, name, measurements):
        super().__init__(name)
        self.measurements = measurements
        self.index = 0

    def measure(self, environment=None, state=None):
        if self.index < len(self.measurements):
            self.current_measurement = self.measurements[self.index]
            self.index += 1
        return self.current_measurement
    
@dataclass
class MockEnvironmnent():
    sensors: dict[str, Sensor]

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

@pytest.fixture
def simulation_state():
    return SimulationState(
        simulation_start_datetime=pd.Timestamp("2026-01-01 00:00:00"),
        time=0,
        time_step=900  # 15 minutes
    )

@pytest.fixture
def init_context(simulation_state):
    # Create an InitContext with the simulation state
    measurements = np.random.rand(50)  # 50 random measurements
    mock_sensor = MockSensor("test_sensor", measurements)
    mock_environmnent = MockEnvironmnent(sensors = {'test_sensor': mock_sensor})
    mock_sensor.measure(environment=mock_environmnent, state=simulation_state)  # Initialize the first measurement
    return InitContext(environment = mock_environmnent, state = simulation_state)
    

def test_OfflineForecastPredictor_predict(simulation_state, forecast_df):
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


def test_OfflineForecastPredictor_select_issue_time(forecast_df):
    predictor = OfflineForecastPredictor(
        name="test_predictor",
        forecast_df=forecast_df,
        variable_to_predict="DHI"
    )

    now = pd.Timestamp("2026-01-02 12:00:00")
    issue_time = predictor._select_issue_time(now)
    assert issue_time == pd.Timestamp("2026-01-02 00:00:00")


def test_OfflineForecastPredictor_missing_variable(forecast_df, simulation_state):
    predictor = OfflineForecastPredictor(
        name="test_predictor",
        forecast_df=forecast_df,
        variable_to_predict="MISSING_VAR"
    )

    with pytest.raises(KeyError):
        predictor.predict(2, simulation_state)


def test_DailyProfilePredictor_predict():
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


def test_DailyProfilePredictor_invalid_profile():
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


@pytest.fixture
def ann_predictor(init_context):
    predictor = ANNBasedPredictor(
        prediction_horizon_h=1,  # 1 hour
        sensor_name="test_sensor",
        window_size_h=24,  # 24 hours
        retrain_interval_h=24,  # 24 hours
        min_sample_size_h=48,  # 24 hours
        hidden_layer_sizes=(10,),
        max_iter=1000
    )
    predictor.initialize(init_context)
    return predictor


def test_ann_based_predictor_initialization(ann_predictor):
    assert ann_predictor.prediction_horizon_h == 1
    assert ann_predictor.sensor_name == "test_sensor"
    assert ann_predictor.window_size_h == 24
    assert ann_predictor.retrain_interval_h == 24
    assert ann_predictor.min_sample_size_h == 48
    assert ann_predictor.is_trained is False

def test_ann_based_predictor_initialized_with_wrong_sizes():
    with pytest.raises(ValueError):
        predictor = ANNBasedPredictor(
            prediction_horizon_h = 4,  # 1 hour
            sensor_name = "test_sensor",
            window_size_h = 24,  # 24 hours
            retrain_interval_h = 24,  # 24 hours
            min_sample_size_h = 25,  # 24 hours
            hidden_layer_sizes = (10,),
            max_iter = 1000
        )

def test_ann_based_predictor_update(ann_predictor):
    # Add some data to the buffer
    for i in range(10):
        ann_predictor.update(i * 900)  # i hours in seconds

    assert len(ann_predictor.buffer) == 10
    assert len(ann_predictor.time_buffer) == 10

def test_ann_based_predictor_train(ann_predictor):
    # Add enough data to trigger training
    for i in range(50*4):  # 50 hours
        ann_predictor.update(i * 900)

    # Check if the model is trained
    assert ann_predictor.is_trained is True

def test_ann_based_predictor_predict_not_trained(ann_predictor):
    # Add some data but not enough to trigger training
    for i in range(10):
        ann_predictor.update(i * 900)

    # Make a prediction
    predictions = ann_predictor.predict(1, SimulationState(time=10 * 900))

    # Check if the prediction is a persistence forecast
    assert np.all(predictions == ann_predictor.buffer[-1])

def test_ann_based_predictor_predict_trained(ann_predictor, simulation_state):
    # Add enough data to trigger training
    for i in range(int(80*3600/simulation_state.time_step)):  # 80 hours
        ann_predictor.update(i * simulation_state.time_step)

    # Make a prediction
    predictions = ann_predictor.predict(1, SimulationState(time=80 * 3600))

    # Check if the prediction is an array of the correct length
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == ann_predictor.prediction_horizon

def test_ann_based_predictor_predict_wrong_horizon(ann_predictor, simulation_state):
    # Add enough data to trigger training
    for i in range(int(80*3600/simulation_state.time_step)):  # 25 hours
        ann_predictor.update(i * 3600)

    # Make a prediction with a wrong horizon
    with pytest.raises(ValueError):
        ann_predictor.predict(2, SimulationState(time=25 * 3600))

def test_ann_based_predictor_encode_hour(ann_predictor):
    # Test the _encode_hour method
    sin_h, cos_h = ann_predictor._encode_hour(0)  # Midnight
    assert np.isclose(sin_h, 0)
    assert np.isclose(cos_h, 1)
    sin_h, cos_h = ann_predictor._encode_hour(6 * 3600)  # 6 AM
    assert np.isclose(sin_h, 1)
    assert np.isclose(cos_h, 0)
    sin_h, cos_h = ann_predictor._encode_hour(12 * 3600)  # Noon
    assert np.isclose(sin_h, 0)
    assert np.isclose(cos_h, -1)
    sin_h, cos_h = ann_predictor._encode_hour(18 * 3600)  # 6 PM
    assert np.isclose(sin_h, -1)
    assert np.isclose(cos_h, 0)
    sin_h, cos_h = ann_predictor._encode_hour(42 * 3600)  # 6 PM
    assert np.isclose(sin_h, -1)
    assert np.isclose(cos_h, 0)

def test_ann_based_predictor_prepare_data(ann_predictor, simulation_state):
    # Add some data to the buffer
    for i in range(int(80*3600/simulation_state.time_step)):  # 50 hours
        ann_predictor.update(i * 3600)

    # Prepare the data for training
    X, Y = ann_predictor._prepare_data()

    # Check if the data is prepared correctly
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert len(X) == len(Y)
    assert X.shape[1] == ann_predictor.window_size + 2  # window_size + sin_h + cos_h
    assert Y.shape[1] == ann_predictor.prediction_horizon