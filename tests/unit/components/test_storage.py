# tests/unit/test_components_battery.py
import numpy as np
import pytest

def test_battery_creation():
    from energy_system_control import Battery, ElectricalStorageNode
    test_battery = Battery('test_battery', 5.0)
    # Check correct creation and default values
    assert test_battery.max_capacity['test_battery_electrical_node'] == 5.0*3600
    assert test_battery.electrical_node == 'test_battery_electrical_node'
    # Test node creation
    test_battery_node = test_battery.create_storage_nodes()
    assert isinstance(test_battery_node['test_battery_electrical_node'], ElectricalStorageNode)

def test_battery_soc():
    from energy_system_control import Battery, Environment, ConstantPowerProducer
    components = {
        'test_battery': Battery('test_battery', 5.0),
        'producer': ConstantPowerProducer('producer', nodes = ['test_battery_electrical_node'], power = {'test_battery_electrical_node': 1.0})
    }
    env = Environment(components=components)  # dt = 60 s
    time_step = 0.25
    env.run(time_step = time_step, time_end = 1.0)  # simulate 1 hour
    assert env.components['test_battery'].SOC == 0.684


def test_battery_soc_bound_max():
    from energy_system_control import Battery, Environment, ConstantPowerProducer
    components = {
        'test_battery': Battery('test_battery', 5.0),
        'producer': ConstantPowerProducer('producer', nodes = ['test_battery_electrical_node'], power = {'test_battery_electrical_node': 1.0})
    }
    env = Environment(components=components)  # dt = 60 s
    time_step = 0.25
    # Testing warning for SOC too high
    expected_warning_message = 'Storage unit test_battery has storage level higher than maximum allowed at time step 8100.0. Observed SOC is 0.914 while max value is 0.9'
    with pytest.warns(UserWarning, match=expected_warning_message):
        env.run(time_step = time_step, time_end = 10.0)  # simulate 6 hours
    assert True

def test_battery_soc_bound_min():
    from energy_system_control import Battery, Environment, ConstantPowerDemand
    components = {
        'test_battery': Battery('test_battery', 5.0),
        'producer': ConstantPowerDemand('producer', nodes = ['test_battery_electrical_node'], power = {'test_battery_electrical_node': 1.0})
    }
    env = Environment(components=components)  # dt = 60 s
    time_step = 0.25
    # Testing warning for SOC too high
    expected_warning_message = 'Storage unit test_battery has storage level lower than minimum allowed at time step 3600.0. Observed SOC is 0.288 while min value is 0.3'
    with pytest.warns(UserWarning, match=expected_warning_message):
        env.run(time_step = time_step, time_end = 10.0)  # simulate 6 hours
    assert True