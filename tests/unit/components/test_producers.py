# tests/unit/test_components_pvpanel.py
import numpy as np
import pandas as pd
from math import isclose

def test_panel_with_custom_data():
    from energy_system_control import PVpanel
    raw_data = pd.Series(index = [pd.to_datetime('2023-04-13 06:00') + pd.Timedelta(hours=x) for x in [0, 1, 2, 3, 4]], data = [0.12, 0.14, 0.16, 0.13, 0.11])
    pv = PVpanel(
        name = 'test_panel', 
        electrical_node = 'test_PV_node', 
        installed_power = 2,
        raw_data=raw_data
    )
    # Resampling the data to the required time step
    pv.time_step = 0.25 * 3600
    pv.resample_data(pv.time_step, sim_end = 24*3600)
    # Checking input
    assert pv.data[0] == 0.12
    # Testing taking step
    test_step = pv.step()
    assert test_step['test_PV_node'] == 0.25 * 3600 * 0.12 * 2  # Energy generated during the time step in kJ. time_step[h] * hours_to_seconds[s/h] * capacity_factor[-] * installed_power[kW] 
    # Testing the check data function
    pv.check_data()
    assert True

def test_panel_from_PVGIS():
    from energy_system_control import PVpanelFromPVGIS
    pv = PVpanelFromPVGIS(
        name = 'test_panel', 
        electrical_node = 'test_PV_node', 
        installed_power = 2,
        latitude = 44.530,
        longitude = 11.293,
        tilt = 23,
        azimuth = 45
    )
    # Resampling the data to the required time step
    pv.time_step = 0.25 * 3600
    pv.resample_data(time_step = pv.time_step, sim_end = 24*3600)
    # Checking input
    assert isclose(pv.data[50], 0.223, abs_tol=0.01)
    # Testing taking step
    test_step = pv.step()
    assert test_step['test_PV_node'] == 0.0
    # Testing the check data function
    pv.check_data()
    assert True