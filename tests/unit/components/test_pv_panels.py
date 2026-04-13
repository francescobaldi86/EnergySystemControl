# tests/unit/test_components_pvpanel.py
import numpy as np
import pandas as pd
from math import isclose
from energy_system_control.components.explicit_components.pv_panels import PVpanel, PVpanelFromData, PVpanelFromPVGISData, PVpanelFromPVGIS
from energy_system_control.components.base import TimeSeriesData
from energy_system_control.sim.state import SimulationState
from datetime import datetime, timedelta
import os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Base test for instantiation
def test_pv_panel_instantiation():
    ts = TimeSeriesData(raw=pd.Series([1000, 2000], index=[datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0)]), var_type='power', var_unit='W')
    pv_panel = PVpanel(name="test_pv_panel", ts=ts)
    assert isinstance(pv_panel, PVpanel)

    # Test PVpanelFromData instantiation
    data_path = os.path.join(__TEST__, 'DATA')
    filename = "pvgis_data.csv"
    col_name = 'P'
    skip_footer = 11
    pv_panel_from_data = PVpanelFromData(name="test_pv_panel_from_data", data_path=data_path, filename=filename, column_name=col_name, skipfooter=skip_footer)
    assert isinstance(pv_panel_from_data, PVpanelFromData)

    # Test PVpanelFromPVGISData instantiation
    pv_panel_from_pvgis_data = PVpanelFromPVGISData(name="test_pv_panel_from_pvgis_data", data_path=data_path, filename=filename)
    assert isinstance(pv_panel_from_pvgis_data, PVpanelFromPVGISData)

    # Test PVpanelFromPVGIS instantiation
    pv_panel_from_pvgis = PVpanelFromPVGIS(name="test_pv_panel_from_pvgis", installed_power=1000, latitude=45, longitude=8, tilt=30, azimuth=0)
    assert isinstance(pv_panel_from_pvgis, PVpanelFromPVGIS)

def test_panel_with_custom_data():
    from energy_system_control import PVpanel
    from energy_system_control.components.base import TimeSeriesData
    raw_data = pd.Series(index = [pd.to_datetime('2023-04-13 06:00') + pd.Timedelta(hours=x) for x in [0, 1, 2, 3, 4]], data = [0.12, 0.14, 0.16, 0.13, 0.11])
    pv = PVpanel(
        name = 'test_panel',  
        ts=TimeSeriesData(raw = raw_data, var_type = 'power', var_unit = 'kW')
    )
    time_step = 1800
    state = SimulationState(time_id = 0, time_step = time_step)
    # Resampling the data to the required time step
    pv.resample_data(time_step_h = time_step/3600, sim_end_h = 24)
    pv.create_ports()
    # Checking input
    assert isclose(pv.ts.data[0], 0.12, abs_tol = 0.01)
    # Testing taking step
    pv.step(state)
    assert pv.ports[pv.port_name].flow['electricity'] == -time_step * 0.12  # Energy generated during the time step in kJ

def test_panel_from_PVGIS():
    from energy_system_control import PVpanelFromPVGIS
    pv = PVpanelFromPVGIS(
        name = 'test_panel', 
        installed_power = 2,
        latitude = 44.530,
        longitude = 11.293,
        tilt = 23,
        azimuth = 45
    )
    time_step = 900
    # Resampling the data to the required time step
    pv.resample_data(time_step_h = time_step/3600, sim_end_h = 24)
    pv.create_ports()
    # Checking input
    assert isclose(pv.ts.data[50], 0.446, abs_tol=0.01)
    # Testing taking step
    state = SimulationState(time_id = 0, time_step = time_step)
    pv.step(state)
    assert pv.ports[pv.port_name].flow['electricity'] == 0.0
    # Testing the check data function
    # pv.check_data()

def test_pv_panel_load_data_from_csv_file():
    from energy_system_control import PVpanelFromData
    test_pv = PVpanelFromData(
        name = 'test_pv',
        data_path = os.path.join(__TEST__, 'DATA'),
        filename = 'pvgis_data.csv', 
        column_name = 'P', 
        skipfooter=11
    )
    time_step = 900
    # Resampling the data to the required time step
    test_pv.resample_data(time_step_h = time_step/3600, sim_end_h = 24)
    test_pv.create_ports()

def test_pv_panel_load_data_from_csv_pvgis_file():
    test_pv = PVpanelFromPVGISData(
        name = 'test_pv',
        data_path = os.path.join(__TEST__, 'DATA'),
        filename = 'pvgis_data.csv' 
    )
    time_step = 900
    # Resampling the data to the required time step
    test_pv.resample_data(time_step_h = time_step/3600, sim_end_h = 24)
    test_pv.create_ports()