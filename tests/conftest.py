# tests/conftest.py
import json
import pathlib
import numpy as np
import pandas as pd
import pytest

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
