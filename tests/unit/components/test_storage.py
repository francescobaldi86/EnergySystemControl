# tests/unit/test_components_battery.py
import numpy as np
import pytest, math

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

def test_multinode_water_tank_creation():
    from energy_system_control import MultiNodeHotWaterTank
    test_storage = MultiNodeHotWaterTank(name = 'test_tank',
                                         volume = 250,
                                         heat_injection_nodes={'heat_pump_condenser_node': 2},
                                         height = 1.2)
    assert test_storage.height == 1.2
    assert test_storage.T_0 == 40.0 + 273.15
    assert len(test_storage.T_layer) == 5
    assert test_storage.T_layer[0] > test_storage.T_layer[test_storage.number_of_layers-1]
    assert math.isclose(test_storage.layer_mass, 250/5, abs_tol=1)

def test_multinode_water_tank_internal_heating_flows():
    from energy_system_control import MultiNodeHotWaterTank
    test_storage = MultiNodeHotWaterTank(name = 'test_tank',
                                         volume = 250,
                                         heat_injection_nodes={'heat_pump_condenser_node': 2},
                                         height = 1.2)
    test_flows = test_storage.calculate_heat_exchange_between_layers()
    assert True

def test_multinode_water_tank_base_system(base_test_env_TES):
    from energy_system_control import MultiNodeHotWaterTank, Environment
    base_test_env_TES['components'].append(MultiNodeHotWaterTank(name = 'test_TES',
                                         volume = 250,
                                         number_of_layers = 10,
                                         heat_injection_nodes = {'heat_pump': 7, 'resistance': 6},
                                         height = 1.2,
                                         T_0 = 80))
    
    env = Environment(nodes = base_test_env_TES["nodes"], components = base_test_env_TES["components"], controllers = base_test_env_TES["controllers"], sensors = base_test_env_TES["sensors"])  # dt = 60 s
    time_step = 0.1
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 24 hours
    df_nodes, df_comps = env.to_dataframe()
    df_nodes.to_csv("C:\\Users\\francesco.baldi\\OneDrive - enea.it\\Documents\\Software\\Python\\EnergySystemControl\\tests\\PLAYGROUND\\test_storage.csv", sep = ";")
    assert True
    

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

@pytest.fixture
def base_test_env_TES():
    from energy_system_control import ElectricalNode, IEAHotWaterDemand, HeatPumpConstantEfficiency, SimplifiedHeatSource, BalancingUtility, ColdWaterGrid, HeaterControllerWithBandwidth, TemperatureSensor
    nodes = [ElectricalNode("contatore")]
    components = [
        IEAHotWaterDemand(name= "demand_DHW", thermal_node = "test_TES_hot_water_outlet_thermal_node", mass_node = "test_TES_hot_water_outlet_mass_node", reference_temperature = 40, profile_name='M'),
        BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        ColdWaterGrid(name = 'water_grid', nodes = ['test_TES_cold_water_inlet_thermal_node', 'test_TES_cold_water_inlet_mass_node']),
        HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "test_TES_heat_pump_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2)
    ]
    controllers = [
        HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor_hp', 40, 10)
    ]
    sensors = [
        TemperatureSensor('storage_tank_temperature_sensor_hp', node_name = 'test_TES_heat_pump_thermal_node')
    ]
    return {'nodes': nodes, 'components': components, 'controllers': controllers, 'sensors': sensors}