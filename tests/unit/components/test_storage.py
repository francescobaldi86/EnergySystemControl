# tests/unit/test_components_battery.py
import numpy as np

def test_battery_creation():
    from energy_system_control import Battery, ElectricalStorageNode
    test_battery = Battery('test_battery', 5.0)
    # Check correct creation and default values
    assert test_battery.max_capacity['test_battery_electrical_node'] == 5.0*3600
    assert test_battery.electrical_node == 'test_battery_electrical_node'
    # Test node creation
    test_battery_node = test_battery.create_storage_nodes()
    assert isinstance(test_battery_node['test_battery_electrical_node'], ElectricalStorageNode)

def test_battery_soc_bounds():
    from energy_system_control import Battery, Environment, ConstantPowerProducer
    components = {
        'test_battery': Battery('test_battery', 5.0),
        'producer': ConstantPowerProducer('producer', nodes = ['test_battery_electrical_node'], power = {'test_battery_electrical_node': 1.0})
    }
    env = Environment(components=components)  # dt = 60 s
    time_step = 0.25
    env.run(time_step = time_step, time_end = 1.0)  # simulate 6 hours
    assert env.components['test_battery'].SOC == 0.23


def test_battery_roundtrip_losses():
    from energy_system_control import Battery
    bat = Battery(capacity_kwh=10, max_charge_kw=5, max_discharge_kw=5, eta_charge=0.9, eta_discharge=0.9)
    bat.set_soc(0.2)
    e_in = bat.charge_energy_kwh(5.0)   # returns accepted kWh
    e_out = bat.discharge_energy_kwh(5.0)
    assert e_out < e_in  # losses present
