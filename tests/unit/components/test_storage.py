# tests/unit/test_components_battery.py
import numpy as np
import pytest

def test_battery_creation():
    from energy_system_control import LithiumIonBattery, ElectricalStorageNode
    test_battery = LithiumIonBattery('test_battery', 5.0)
    # Check correct creation and default values
    assert test_battery.max_capacity['test_battery_electrical_node'] == 5.0*3600
    assert test_battery.internal_node_name == 'test_battery_electrical_node'
    # Test node creation
    test_battery_node = test_battery.create_storage_nodes()
    assert isinstance(test_battery_node['test_battery_electrical_node'], ElectricalStorageNode)

def test_battery_soc(base_test_env):
    time_step = 0.25
    base_test_env.run(time_step = time_step, time_end = 1.0)  # simulate 1 hour
    assert base_test_env.components['test_battery'].SOC == 0.684


@pytest.fixture
def base_test_env():
    from energy_system_control import LithiumIonBattery, Environment, ConstantPowerProducer, ElectricalNode, Inverter, SOCSensor, PowerBalanceSensor
    nodes = {'main_node': ElectricalNode('main_node')}
    components = {
        'test_battery': LithiumIonBattery('test_battery', electrical_node='main_node', capacity = 5.0),
        'producer': ConstantPowerProducer('producer', nodes = ['main_node'], power = {'main_node': 1.0})
    }
    sensors = {
        'soc_sensor': SOCSensor('soc_sensor', 'test_battery', 'test_battery_electrical_node'),
        'grid_flow_sensor': PowerBalanceSensor('grid_flow_sensor', 'main_node'),
    }
    controllers = {'inverter': Inverter('inverter', 'test_battery', 'soc_sensor', 'grid_flow_sensor')}
    env = Environment(nodes = nodes, components=components, sensors=sensors, controllers=controllers)  # dt = 60 s
    return env