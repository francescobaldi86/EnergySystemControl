# tests/unit/test_components_battery.py
import numpy as np

def test_battery_soc_bounds():
    from energy_system_control import Battery
    bat = Battery(capacity_kwh=5, max_charge_kw=2, max_discharge_kw=2, eta_charge=0.95, eta_discharge=0.95)
    bat.set_soc(0.5)
    bat.charge_kw(100)  # over-ask
    assert 0.0 <= bat.soc <= 1.0

def test_battery_roundtrip_losses():
    from energy_system_control import Battery
    bat = Battery(capacity_kwh=10, max_charge_kw=5, max_discharge_kw=5, eta_charge=0.9, eta_discharge=0.9)
    bat.set_soc(0.2)
    e_in = bat.charge_energy_kwh(5.0)   # returns accepted kWh
    e_out = bat.discharge_energy_kwh(5.0)
    assert e_out < e_in  # losses present
