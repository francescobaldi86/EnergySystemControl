# tests/unit/test_components_battery.py
import numpy as np
import pytest, math
from energy_system_control.core.base_classes import InitContext

def test_battery_creation():
    from energy_system_control import LithiumIonBattery
    from energy_system_control.core.ports import ElectricPort
    test_battery = LithiumIonBattery('test_battery', 5.0)
    # Check correct creation and default values
    assert test_battery.max_capacity == 5.0 * 3600
    assert test_battery.port_name == 'test_battery_electricity_port'
    # Test node creation
    test_battery.create_ports()
    assert isinstance(test_battery.ports[test_battery.port_name], ElectricPort)
    assert True

def test_battery_soc(base_test_env):
    from energy_system_control import SimulationConfig, Simulator
    sim_config = SimulationConfig(time_start_h = 0.0, time_end_h = 1, time_step_h = 0.5)
    sim = Simulator(base_test_env, sim_config)
    results = sim.run()
    assert math.isclose(base_test_env.components['battery'].SOC, 0.684, abs_tol=0.01)

def test_multinode_water_tank_creation():
    from energy_system_control import MultiNodeHotWaterTank
    from energy_system_control.sim.state import SimulationState
    test_storage = MultiNodeHotWaterTank(name = 'test_tank',
                                         tank_volume = 250,
                                         height_main_heat_input = 0.2,
                                         tank_height = 1.2)
    assert test_storage.height == 1.2
    assert test_storage.T_0 == 40.0 + 273.15
    test_storage.create_ports()
    ctx = InitContext(environment = None, state = SimulationState(time = 0.0, time_step = 900))
    test_storage.initialize(ctx)
    assert len(test_storage.T_layer) == 5
    assert test_storage.T_layer[0] > test_storage.T_layer[test_storage.number_of_layers-1]
    assert math.isclose(test_storage.layer_mass, 250/5, abs_tol=1)

def test_multinode_water_tank_internal_heating_flows():
    from energy_system_control import MultiNodeHotWaterTank
    from energy_system_control.sim.state import SimulationState
    test_storage = MultiNodeHotWaterTank(name = 'test_tank',
                                         tank_volume = 250,
                                         height_main_heat_input = 0.2,
                                         tank_height = 1.2)
    test_storage.create_ports()
    ctx = InitContext(environment = None, state = SimulationState(time = 0.0, time_step = 900))
    test_storage.initialize(ctx)
    test_flows = test_storage.calculate_heat_exchange_between_layers()
    assert True

def test_multinode_water_tank_base_system(base_test_env_TES):
    from energy_system_control.sim.config import SimulationConfig
    from energy_system_control.sim.simulator import Simulator
    time_step = 0.1
    sim_config = SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = time_step)
    sim = Simulator(base_test_env_TES, sim_config)
    results = sim.run()
    assert True
    

@pytest.fixture
def base_test_env():
    from energy_system_control import LithiumIonBattery, Environment, ConstantPowerProducer, InverterController, SOCSensor, Inverter, BalancingUtility, Simulator, SimulationConfig
    components = [
        LithiumIonBattery('battery', capacity = 5.0, SOC_0 = 0.5),
        ConstantPowerProducer('producer', production_type='electricity', power = 1.0),
        Inverter(name = 'inverter'),
        BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
    ]
    sensors = [
        SOCSensor('battery_SOC_sensor', 'battery'),
    ]
    controllers = [InverterController('inverter_controller', 'inverter', 'battery')]
    connections = [
        ('inverter_PV_input_port', 'producer_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port')
    ]
    env = Environment(components=components, sensors=sensors, controllers=controllers, connections=connections)  # dt = 60 s
    
    return env

@pytest.fixture
def base_test_env_TES():
    from energy_system_control import MultiNodeHotWaterTank, IEAHotWaterDemand, HeatPumpConstantEfficiency, TankTemperatureSensor, BalancingUtility, ColdWaterGrid, HeaterControllerWithBandwidth, Environment
    components = [
        IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2),
        MultiNodeHotWaterTank(name = 'test_TES', max_temperature = 80, tank_volume = 250, number_of_layers=10, tank_height=1.5, height_main_heat_input=0.5, height_aux_heat_input=0.3, T_0=80),
        BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        ColdWaterGrid(name = 'water_grid', utility_type = 'fluid')
    ]
    controllers = [
        HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10)
    ]
    sensors = [
        TankTemperatureSensor('storage_tank_temperature_sensor', 'test_TES')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'test_TES_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'test_TES_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'electric_grid_electricity_port'),
        ('test_TES_cold_water_input_port', 'water_grid_fluid_port')
    ]
    env = Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    return env