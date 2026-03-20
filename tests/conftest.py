# tests/conftest.py
import json, pathlib, os, pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext
from tests.utils import MockSensor, MockEnvironment

@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def weather_small(data_dir):
    # columns: time, ghi, temp_air, wind, etc.
    return pd.read_csv(data_dir / "weather_small.csv", parse_dates=["time"]).set_index("time")

@pytest.fixture(scope="session")
def load_small(data_dir):
    # columns: time, p_load (W), q_load or thermal_load if needed
    return pd.read_csv(data_dir / "load_small.csv", parse_dates=["time"]).set_index("time")

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def time_index(weather_small):
    return weather_small.index

@pytest.fixture
def minimal_environment(weather_small, load_small):
    """Build a tiny end-to-end system using your public API only."""
    # adjust imports to your actual public API names
    from energy_system_control import Environment, PVPanel, Battery, ThermalNode

    env = Environment(time_index=weather_small.index, ambient_temp=weather_small["temp_air"])
    pv = PVPanel(area=10.0, efficiency=0.20)  # example
    bat = Battery(capacity_kwh=5.0, max_charge_kw=2.5, max_discharge_kw=2.5, eta_charge=0.95, eta_discharge=0.95)

    # Create nodes / connect components as your API requires
    elec = ThermalNode()  # or ElectricalNode if you have it; adjust to your types

    # env.add_component(pv, node=elec) ...
    # env.add_component(bat, node=elec) ...
    # env.add_demand(load_small["p_load"], node=elec) ...

    return env




@pytest.fixture
def mock_init_context(request):
    measurements = getattr(request, "param", [0.0])  # default if nothing provided
    mock_sensor = MockSensor("test_sensor", measurements)
    sim_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=900  # 15 minutes
        )
    mock_env = MockEnvironment(sensors={'test_sensor': mock_sensor})
    init_ctx = InitContext(environment=mock_env, state=sim_state)
    mock_sensor.initialize(init_ctx)  # Initialize the sensor to set the first measurement
    return init_ctx
    


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
def init_context_autocorr(simulation_state):
    """Create a mock initialization context for AutocorrPredictor tests."""
    # Create synthetic predictable measurements
    measurements = np.array([1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0, 0.5, 1.0, 1.5, 
                             2.0, 2.5, 2.0, 1.5, 1.0, 0.5] * 5)  # 80 samples
    mock_sensor = MockSensor("test_sensor", measurements)
    mock_env = MockEnvironment(sensors={'test_sensor': mock_sensor})
    mock_sensor.measure(environment=mock_env, state=simulation_state)
    return InitContext(environment=mock_env, state=simulation_state)

@pytest.fixture
def init_context_ml(simulation_state):
    """Create a mock initialization context for ML-based predictors tests."""
    # Create synthetic predictable measurements
    measurements = np.array([1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0, 0.5, 1.0, 1.5, 
                             2.0, 2.5, 2.0, 1.5, 1.0, 0.5] * 5)  # 80 samples
    mock_sensor = MockSensor("test_sensor", measurements)
    mock_env = MockEnvironment(sensors={'test_sensor': mock_sensor})
    mock_sensor.measure(environment=mock_env, state=simulation_state)
    return InitContext(environment=mock_env, state=simulation_state)

@pytest.fixture
def __TEST__():
    # This fixture can be used to store any test-specific data or state
    return os.path.dirname(__file__)

# ============================================================================
# DHW Demand Prediction Tests with Varying Dataset Sizes
# ============================================================================

@pytest.fixture
def dhw_demand_data(__TEST__):
    """Load and prepare DHW demand data from the test data file."""
    import os
    
    # Get the path to the data file
    data_file = os.path.join(__TEST__, 'DATA', 'dhw_demand_data.csv')
    
    # Load the data
    df = pd.read_csv(data_file, sep=';')
    
    # Extract demand values as a numpy array
    demand_values = df['DHW demand'].values
    
    return demand_values


