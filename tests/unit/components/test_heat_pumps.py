import pytest
import energy_system_control as esc
import math
from energy_system_control.helpers import C2K

  

def test_abstract_class_error():
    assert True


def test_fixed_efficiency_heat_pump_creation():
    assert True


def test_system_with_fixed_efficiecy_heat_pump(base_environment_info):
    base_environment_info['components'].append(esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2))
    base_environment_info['connections'] +=[
        ('hot_water_storage_main_heat_input_port', 'heat_pump_heat_output_port'),
        ('heat_pump_electricity_input_port', 'electric_grid_electricity_port')]
    env = esc.Environment(
        components = base_environment_info["components"], 
        controllers = base_environment_info["controllers"], 
        sensors = base_environment_info["sensors"], 
        connections = base_environment_info["connections"])  # dt = 60 s
    sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = 0.1)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 323, abs_tol = 1)
    assert math.isclose(results.get_cumulated_electricity('heat_pump_electricity_input_port'), 15.8, abs_tol = 0.1)

def test_lorentz_heat_pump_creation():
    # Test creation from COP design, in baseline conditions
    hp = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2)
    assert math.isclose(hp.Wdot_design, 1.5 / 3.2, abs_tol = 0.001)
    assert math.isclose(hp.eta_lorentz, 0.433, abs_tol = 0.001)
    # Creation from COP design, changing temperature using design data from reference Immergas HP
    hp_immergas = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.3, COP_design = 3.0, T_water_design = 50)
    assert math.isclose(hp_immergas.eta_lorentz, 0.485, abs_tol = 0.001)
    # Creation from COP design, changing temperature using design data from reference Bosch HP. Also testing off design conditions
    hp_bosch = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.3, COP_design = 2.5*1.36, T_water_design = 46)
    assert math.isclose(hp_bosch.eta_lorentz, 0.514, abs_tol = 0.001)
    assert math.isclose(hp_bosch._get_efficiency(C2K(14), C2K(46)), 3.97, abs_tol = 0.01)
    assert math.isclose(hp_bosch._get_efficiency(C2K(2), C2K(46)), 3.08, abs_tol = 0.01)
    # Test creation from Lorentz efficiency
    hp_bosch = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.3, eta_lorentz = 0.514, T_water_design = 46)
    assert math.isclose(hp_bosch.COP_design, 2.5*1.36, abs_tol = 0.001)

def test_lorentz_heat_pump_variable_heat_output():
    # Tests the possibility of having the heat output variable with temperatures
    hp = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2, heat_capacity_loss=0.01)
    assert hp._get_heat_output(C2K(0), C2K(50)) < hp._get_heat_output(C2K(0), C2K(40))
    assert hp._get_heat_output(C2K(0), C2K(50)) < hp._get_heat_output(C2K(10), C2K(50))

def test_system_with_lorentz_heat_pump_from_COP(base_environment_info):
    base_environment_info['components'].append(esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2))
    base_environment_info['connections'] +=[
        ('hot_water_storage_main_heat_input_port', 'heat_pump_heat_output_port'),
        ('heat_pump_electricity_input_port', 'electric_grid_electricity_port')]
    env = esc.Environment(
        components = base_environment_info["components"], 
        controllers = base_environment_info["controllers"], 
        sensors = base_environment_info["sensors"], 
        connections = base_environment_info["connections"])  # dt = 60 s
    sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = 0.1)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    assert math.isclose(env.components['heat_pump'].eta_lorentz, 0.433, abs_tol = 0.01)
    assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 323, abs_tol = 1)
    assert math.isclose(results.get_cumulated_electricity('heat_pump_electricity_input_port'), 12.2, abs_tol = 0.1)



@pytest.fixture
def base_environment_info():
    # One node: house thermal mass
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        esc.ElectricityGrid(name = 'electric_grid'),
       esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10)
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', component_name = 'hot_water_storage')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
    ]
    return {'components': components, 'controllers': controllers, 'sensors': sensors, 'connections': connections}
    
    