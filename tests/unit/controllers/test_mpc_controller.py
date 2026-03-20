# test_mpc_controller.py
import pytest
import energy_system_control as esc
from energy_system_control.controllers.MPC import MPCController, MPCController_HybridDHW
from energy_system_control.controllers.predictors import PerfectTimeSeriesPredictor, ProfileARPredictor
import math, os
import pandas as pd

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class MockMPCController(MPCController):
    """Concrete implementation of MPCController for testing purposes."""
    def __init__(self, name, controlled_components, sensors, horizon, solver):
        super().__init__(name, controlled_components, sensors, horizon, solver)

    def get_action(self, state):
        # Implement the get_action method
        pass

    def initialize(self):
        # Implement the initialize method
        pass

@pytest.fixture
def mock_controller():
    return MockMPCController(
        name="test_controller",
        controlled_components=["component1", "component2"],
        sensors={"sensor1": "value1", "sensor2": "value2"},
        horizon=10.0,
        solver='OSQP'
    )

def test_initialization(mock_controller):
    assert mock_controller.name == "test_controller"
    assert mock_controller.controlled_component_names == ["component1", "component2"]
    assert mock_controller.sensor_names == {"sensor1": "value1", "sensor2": "value2"}
    assert mock_controller.horizon == 10.0
    assert mock_controller.solver == 'OSQP'

def test_get_obs(mock_controller):
    # You'll need to create a mock environment and state for this test
    # obs = mock_controller.get_obs(environment, state)
    # Add assertions based on the expected behavior of get_obs
    assert True

def test_initialize(mock_controller):
    result = mock_controller.initialize()
    assert result is None

def test_horizon_validation():
    with pytest.raises(ValueError):
        MockMPCController(
            name="test_controller",
            controlled_components=["component1", "component2"],
            sensors={"sensor1": "value1", "sensor2": "value2"},
            horizon=0.0,
            solver='HIGHS'
        )


def test_MPC_HybridDHW_application(test_components, test_sensors):
    # Test of a full system
    controllers = [
        MPCController_HybridDHW('MPC_controller',
                                storage_temperature_sensor = 'storage_tank_temperature_sensor',
                                battery_SOC_sensor = 'battery_SOC_sensor',
                                PV_power_predictor_name = 'pv_power_predictor',
                                heat_demand_predictor_name = 'dhw_demand_predictor',
                                horizon = 24),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
                ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port')
    ]
    predictors = [PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'), 
                  PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')]
    # Create environment
    env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections, predictors=predictors)  # dt = 60 s
    # Create simulator object
    sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = 1.0)
    sim = esc.Simulator(env, sim_config)
    # Run simulation
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()    
    heat_pump_energy_demand = results.get_cumulated_electricity('heat_pump_electricity_input_port')
    electricity_from_pv = results.get_cumulated_electricity('inverter_PV_input_port')
    net_electricity_demand = results.get_cumulated_electricity('electric_grid_electricity_port')
    electricity_to_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only positive')
    electricity_from_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only negative')
    assert math.isclose(electricity_from_pv, 29, abs_tol = 2)
    assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    assert math.isclose(net_electricity_demand, 12, abs_tol = 2)
    assert math.isclose(electricity_from_grid, 2, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 13, abs_tol = 2)
    assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 315, abs_tol = 1)


def test_MPC_with_hybrid_AR_predictors(test_components, test_sensors):
    """
    Test MPC controller using ANN-based predictors for PV power and heat demand.
    
    This test verifies that:
    1. The MPC controller can work with ANN-based predictors
    2. The simulation runs successfully with ANN predictions
    3. The results are reasonable (grid electricity, heat pump energy, etc.)
    """
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port')
    ]
    
    controllers = [
        MPCController_HybridDHW('MPC_controller',
                                storage_temperature_sensor='storage_tank_temperature_sensor',
                                battery_SOC_sensor='battery_SOC_sensor',
                                PV_power_predictor_name='pv_power_predictor',
                                heat_demand_predictor_name='dhw_demand_predictor',
                                horizon=24),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
    ]
    
    predictors = [
        ProfileARPredictor(
            prediction_horizon_h=24,
            sensor_name='PV_power_sensor',
            name='pv_power_predictor'
        ),
        ProfileARPredictor(
            prediction_horizon_h=24,
            sensor_name='demand_heat_flow_sensor',
            name='dhw_demand_predictor'
        )
    ]
    
    # Create environment
    env = esc.Environment(
        components=test_components,
        controllers=controllers,
        sensors=test_sensors,
        connections=connections,
        predictors=predictors
    )
    
    # Create simulator object
    sim_config = esc.SimulationConfig(time_start_h=0.0, time_end_h=24.0*7, time_step_h=1.0)
    sim = esc.Simulator(env, sim_config)
    
    # Run simulation
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    
    # Get cumulated electricity values
    heat_pump_energy_demand = results.get_cumulated_electricity('heat_pump_electricity_input_port')
    electricity_from_pv = results.get_cumulated_electricity('inverter_PV_input_port')
    net_electricity_demand = results.get_cumulated_electricity('electric_grid_electricity_port')
    electricity_to_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only positive')
    electricity_from_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only negative')
    
    # Verify that simulation produced reasonable results
    assert heat_pump_energy_demand > 0, "Heat pump should consume energy"
    assert electricity_from_pv > 0, "PV panels should generate electricity"
    assert electricity_from_grid > 0 or electricity_to_grid > 0, "Should have some grid interaction"
    
    # Check that values are in reasonable ranges (allowing more tolerance for ANN than for perfect predictors)
    assert electricity_from_pv < 50, "PV energy should be reasonable"
    assert heat_pump_energy_demand < 30, "Heat pump energy should be reasonable"
    assert abs(net_electricity_demand) < 30, "Total net electricity demand should be reasonable"


def test_MPC_perfect_vs_HybridAR_predictors(test_components, test_sensors):
    """
    Compare MPC controller performance with perfect time series predictors vs ANN-based predictors.
    
    This test verifies that:
    1. Both prediction methods allow successful MPC control
    2. Perfect predictions result in lower grid electricity purchases (as expected)
    3. ANN-based predictions, while less optimal, still provide reasonable control
    """
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port')
    ]
    
    # Simulation parameters
    sim_config = esc.SimulationConfig(time_start_h=0.0, time_end_h=24.0*7, time_step_h=1.0)
    
    # --- Run with Perfect Time Series Predictors ---
    controllers_perfect = [
        MPCController_HybridDHW('MPC_controller',
                                storage_temperature_sensor='storage_tank_temperature_sensor',
                                battery_SOC_sensor='battery_SOC_sensor',
                                PV_power_predictor_name='pv_power_predictor',
                                heat_demand_predictor_name='dhw_demand_predictor',
                                horizon=24),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
    ]
    
    predictors_perfect = [
        PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'),
        PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')
    ]
    
    env_perfect = esc.Environment(
        components=test_components,
        controllers=controllers_perfect,
        sensors=test_sensors,
        connections=connections,
        predictors=predictors_perfect
    )
    
    sim_perfect = esc.Simulator(env_perfect, sim_config)
    results_perfect = sim_perfect.run()
    
    # Get grid electricity values for perfect prediction case
    grid_electricity_perfect = results_perfect.get_cumulated_electricity(
        'electric_grid_electricity_port',
        sign='only negative'
    )
    heat_pump_perfect = results_perfect.get_cumulated_electricity('heat_pump_electricity_input_port')
    pv_perfect = results_perfect.get_cumulated_electricity('inverter_PV_input_port')
    
    # --- Run with ANN-Based Predictors ---
    controllers_ann = [
        MPCController_HybridDHW('MPC_controller',
                                storage_temperature_sensor='storage_tank_temperature_sensor',
                                battery_SOC_sensor='battery_SOC_sensor',
                                PV_power_predictor_name='pv_power_predictor',
                                heat_demand_predictor_name='dhw_demand_predictor',
                                horizon=24),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
    ]
    
    predictors_ar = [
        ProfileARPredictor(
            prediction_horizon_h=24,
            sensor_name='PV_power_sensor',
            name='pv_power_predictor'
        ),
        ProfileARPredictor(
            prediction_horizon_h=24,
            sensor_name='demand_heat_flow_sensor',
            name='dhw_demand_predictor'
        )
    ]
    
    env_ann = esc.Environment(
        components=test_components,
        controllers=controllers_ann,
        sensors=test_sensors,
        connections=connections,
        predictors=predictors_ar
    )
    
    sim_ann = esc.Simulator(env_ann, sim_config)
    results_ann = sim_ann.run()
    
    # Get grid electricity values for ANN prediction case
    grid_electricity_ann = results_ann.get_cumulated_electricity(
        'electric_grid_electricity_port',
        sign='only negative'
    )
    heat_pump_ann = results_ann.get_cumulated_electricity('heat_pump_electricity_input_port')
    pv_ann = results_ann.get_cumulated_electricity('inverter_PV_input_port')
    
    # --- Assertions ---
    # Both models should produce valid results
    assert grid_electricity_perfect > 0, "Perfect predictor case should require some grid electricity"
    assert heat_pump_perfect > 0, "Perfect predictor case should require heat pump"
    assert pv_perfect > 0, "Perfect predictor case should generate PV power"
    
    assert grid_electricity_ann > 0, "ANN predictor case should require some grid electricity"
    assert heat_pump_ann > 0, "ANN predictor case should require heat pump"
    assert pv_ann > 0, "ANN predictor case should generate PV power"
    
    # Perfect predictions should result in lower or equal grid electricity purchases
    # (perfect knowledge allows better optimization)
    assert grid_electricity_ann >= grid_electricity_perfect * 0.9, \
        f"ANN predictions should not perform catastrophically worse (ANN: {grid_electricity_ann}, Perfect: {grid_electricity_perfect})"
    assert grid_electricity_perfect <= grid_electricity_ann, \
        f"Perfect predictions should result in lower or equal grid purchases: Perfect={grid_electricity_perfect}, ANN={grid_electricity_ann}"
    

def test_compare_mpc_to_other_controllers(test_components, test_sensors, test_predictors):
    # Prepare dataframe for comparison
    df_comp = pd.DataFrame(columns = ['baseline', 'rule-based', 'MPC'], index = ['Heat pump', 'From grid', 'To grid'])
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port')
    ]
    controllers_dict = {
        'baseline': [
            esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', temperature_comfort = 40, temperature_bandwidth = 10),
            esc.InverterController('inverter_controller', 'inverter', 'battery')
                ],
        'rule-based': [
            esc.HeatPumpRuleBasedController('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 'PV_power_sensor', temperature_comfort = 40, temperature_bandwidth = 10, power_PV_activation = 0.500),
            esc.InverterController('inverter_controller', 'inverter', 'battery')
                ],
        'MPC': [
            MPCController_HybridDHW('MPC_controller',
                                    storage_temperature_sensor = 'storage_tank_temperature_sensor',
                                    battery_SOC_sensor = 'battery_SOC_sensor',
                                    PV_power_predictor_name = 'pv_power_predictor',
                                    heat_demand_predictor_name = 'dhw_demand_predictor',
                                    horizon = 24),
            esc.InverterController('inverter_controller', 'inverter', 'battery')
                ]
    }
    for control_type in ['baseline', 'rule-based', 'MPC']:
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers_dict[control_type], sensors=test_sensors, connections=connections, predictors=test_predictors)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*3, time_step_h = 0.5)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_comp.loc['Heat pump', control_type] = results.get_cumulated_electricity('heat_pump_electricity_input_port')
        df_comp.loc['To grid', control_type] = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only positive')
        df_comp.loc['From grid', control_type] = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only negative')
    assert df_comp.loc['To grid', 'baseline'] >= df_comp.loc['To grid', 'rule-based']
    assert df_comp.loc['To grid', 'rule-based'] >= df_comp.loc['To grid', 'MPC']
    assert df_comp.loc['From grid', 'baseline'] >= df_comp.loc['From grid', 'rule-based']
    assert df_comp.loc['From grid', 'rule-based'] >= df_comp.loc['From grid', 'MPC']

@pytest.fixture
def test_components():
    test_components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design= 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.ElectricityGrid(name = 'electric_grid', cost_of_electricity_purchased=0.24, value_of_electricity_sold=0.06),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGISData(name = 'pv_panels', data_path=os.path.join(__TEST__, 'DATA'), filename = 'pvgis_data.csv', rescale_factor = 3.0),
        esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter')
    ]
    return test_components

@pytest.fixture
def test_sensors():
    test_sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.ElectricPowerSensor('PV_power_sensor', 'inverter_PV_input_port'),
        esc.HotWaterDemandSensor('demand_heat_flow_sensor', 'demand_DHW')
    ]
    return test_sensors

@pytest.fixture
def test_predictors():
    test_predictors = [PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'), 
                       PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')]
    return test_predictors