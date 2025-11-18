import pytest
import energy_system_control as esc
import math

  

def test_abstract_class_error():
    assert True


def test_fixed_efficiency_heat_pump_creation():
    assert True


def test_system_with_fixed_efficiecy_heat_pump(base_environment_info):
    base_environment_info['components'].append(esc.HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2))
    env = esc.Environment(nodes = base_environment_info["nodes"], components = base_environment_info["components"], controllers = base_environment_info["controllers"], sensors = base_environment_info["sensors"])  # dt = 60 s
    time_step = 0.1
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 24 hours
    df_nodes, df_comps = env.to_dataframe()
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 323, abs_tol = 1)
    assert math.isclose(-df_comps.loc[:, ('heat_pump', 'contatore')].sum() * env.time_step / 3600, 13.7, abs_tol = 0.1)

def test_lorentz_heat_pump_creation():
    # Test creation from COP design, in baseline conditions
    hp = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP_design = 3.2)
    assert math.isclose(hp.Wdot_design, 1.5 / 3.2, abs_tol = 0.001)
    assert math.isclose(hp.eta_lorentz, 0.433, abs_tol = 0.001)
    # Creation from COP design, changing temperature using design data from reference Immergas HP
    hp_immergas = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.3, COP_design = 3.0, T_water_design = 50)
    assert math.isclose(hp_immergas.eta_lorentz, 0.485, abs_tol = 0.001)
    # Creation from COP design, changing temperature using design data from reference Bosch HP. Also testing off design conditions
    hp_bosch = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.3, COP_design = 2.5*1.36, T_water_design = 46)
    assert math.isclose(hp_bosch.eta_lorentz, 0.514, abs_tol = 0.001)
    assert math.isclose(hp_bosch._get_efficiency(14, 46), 3.97, abs_tol = 0.01)
    assert math.isclose(hp_bosch._get_efficiency(2, 46), 3.08, abs_tol = 0.01)
    # Test creation from Lorentz efficiency
    hp_bosch = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.3, eta_lorentz = 0.514, T_water_design = 46)
    assert math.isclose(hp_bosch.COP_design, 2.5*1.36, abs_tol = 0.001)

def test_lorentz_heat_pump_variable_heat_output():
    # Tests the possibility of having the heat output variable with temperatures
    hp = esc.HeatPumpLorentzEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP_design = 3.2, heat_capacity_loss=0.01)
    assert hp._get_heat_output(0, 50) < hp._get_heat_output(0, 40)
    assert hp._get_heat_output(0, 50) < hp._get_heat_output(10, 50)

def test_system_with_lorentz_heat_pump_from_COP(base_environment_info):
    base_environment_info['components'].append(esc.HeatPumpLorentzEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP_design = 3.2))
    env = esc.Environment(nodes = base_environment_info["nodes"], components = base_environment_info["components"], controllers = base_environment_info["controllers"], sensors = base_environment_info["sensors"])  # dt = 60 s
    time_step = 0.1
    assert math.isclose(env.components['heat_pump'].eta_lorentz, 0.433, abs_tol = 0.01)
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 24 hours
    df_nodes, df_comps = env.to_dataframe()
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 323, abs_tol = 1)
    assert math.isclose(-df_comps.loc[:, ('heat_pump', 'contatore')].sum() * env.time_step / 3600, 5.7, abs_tol = 0.1)



@pytest.fixture
def base_environment_info():
    # One node: house thermal mass
    nodes = [esc.ElectricalNode("contatore")]
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        esc.BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        esc.ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10)
    ]
    sensors = [
        esc.TemperatureSensor('storage_tank_temperature_sensor', node_name = 'hot_water_storage_thermal_node')
    ]
    return {'nodes': nodes, 'components': components, 'controllers': controllers, 'sensors': sensors}
    
    