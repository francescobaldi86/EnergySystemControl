import energy_system_control as esc
import pytest, math, os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_1():
    # One node: house thermal mass
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10)
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'electric_grid_electricity_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port')
    ]
    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    time_step = 0.5
    sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = time_step)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 325, abs_tol = 1)
    df_ports.to_csv(os.path.join(__TEST__, 'PLAYGROUND', 'test_1_results_ports.csv'), sep = ";")

def test_2():
    # Testing problem 1 with different time steps. In particular, it verifies that when changing the time step the consumption of the heat pump remains approximately constant
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10)
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'electric_grid_electricity_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port')
    ]
    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    for time_step in [1, 0.5, 0.25, 1/6, 5/60, 1/60]:
    # Test that results remain similar when changing the time step
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*7, time_step_h = time_step)
        sim = esc.Simulator(env, sim_config)
        results = sim.run()  # simulate 6 hours
        heat_pump_energy_demand = results.get_cumulated_electricity('heat_pump_electricity_input_port')
        assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)

def test_3():
    # Trying a more complex system, with PV panels
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.Inverter(name = 'inverter')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        esc.InverterController('inverter_controller', 'inverter', None)
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port')
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
    assert math.isclose(electricity_from_grid, 9, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 22, abs_tol = 2)
    assert math.isclose(net_electricity_demand, 13, abs_tol = 2)
    assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 325, abs_tol = 1)

def test_4():
    # Like test 3, but adding a battery with related controller
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
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