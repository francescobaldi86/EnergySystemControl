# test_mpc_controller.py
import pytest
import energy_system_control as esc
from energy_system_control.controllers.MPC import MPCController, MPCController_HybridDHW
from energy_system_control.controllers.predictors import PerfectTimeSeriesPredictor
import math, os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class TestMPCController(MPCController):
    def __init__(self, name, controlled_components, sensors, predictors, horizon, solver):
        super().__init__(name, controlled_components, sensors, predictors, horizon, solver)

    def get_action(self, state):
        # Implement the get_action method
        pass

    def initialize(self):
        # Implement the initialize method
        pass

@pytest.fixture
def mock_controller():
    return TestMPCController(
        name="test_controller",
        controlled_components=["component1", "component2"],
        sensors={"sensor1": "value1", "sensor2": "value2"},
        predictors={"predictor1": "value1", "predictor2": "value2"},
        horizon=10.0,
        solver='OSQP'
    )

def test_initialization(mock_controller):
    assert mock_controller.name == "test_controller"
    assert mock_controller.controlled_component_names == ["component1", "component2"]
    assert mock_controller.sensor_names == {"sensor1": "value1", "sensor2": "value2"}
    assert mock_controller.predictors == {"predictor1": "value1", "predictor2": "value2"}
    assert mock_controller.horizon == 10.0
    assert mock_controller.solver == 'OSQP'

def test_get_obs(mock_controller):
    # You'll need to create a mock environment and state for this test
    # obs = mock_controller.get_obs(environment, state)
    # Add assertions based on the expected behavior of get_obs
    assert True

def test_initialize(controller):
    result = controller.initialize()
    assert result is None

def test_horizon_validation():
    with pytest.raises(ValueError):
        MPCController(
            name="test_controller",
            controlled_components=["component1", "component2"],
            sensors={"sensor1": "value1", "sensor2": "value2"},
            predictors={"predictor1": "value1", "predictor2": "value2"},
            horizon=0.0,
            solver='HIGHS'
        )


def test_MPC_HybridDHW_application():
    # Test of a full system
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design= 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.ElectricityGrid(name = 'electric_grid', cost_of_electricity_purchased=0.24, value_of_electricity_sold=0.06),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGISData(name = 'pv_panels', installed_power=3.0, data_path=os.path.join(__TEST__, 'DATA'), filename = 'pvgis_data.csv'),
        esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter')
    ]
    controllers = [
        esc.InverterController('inverter_controller', 'inverter', 'battery'),
        MPCController_HybridDHW('MPC_controller',
                                sensors = {'storage_tank_temperature': 'storage_tank_temperature_sensor', 
                                           'battery_SOC': 'battery_SOC_sensor'},
                                PV_power_predictor = PerfectTimeSeriesPredictor('pv_panels'),
                                heat_demand_predictor = PerfectTimeSeriesPredictor('demand_DHW'),
                                horizon = 24)
                ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.SOCSensor('battery_SOC_sensor', 'battery')
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
    # Create environment
    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    # Create simulator object
    sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = 0.5)
    sim = esc.Simulator(env, sim_config)
    # Run simulation
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    # Verify results
    heat_pump_energy_demand = results.get_cumulated_electricity('heat_pump_electricity_input_port')
    electricity_from_pv = results.get_cumulated_electricity('inverter_PV_input_port')
    net_electricity_demand = results.get_cumulated_electricity('electric_grid_electricity_port')
    electricity_to_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only positive')
    electricity_from_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only negative')
    assert math.isclose(electricity_from_pv, 27, abs_tol = 2)
    assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    assert math.isclose(net_electricity_demand, 12, abs_tol = 2)
    assert math.isclose(electricity_from_grid, 2, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 13, abs_tol = 2)
    assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 325, abs_tol = 1)