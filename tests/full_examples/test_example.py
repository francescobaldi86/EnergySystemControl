import energy_system_control as esc
import pytest, math, os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_1():
    # One node: house thermal mass
    nodes = [esc.ElectricalNode("contatore")]
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
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

    env = esc.Environment(nodes=nodes, components=components, controllers = controllers, sensors=sensors)  # dt = 60 s
    time_step = 0.5
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 325, abs_tol = 1)

def test_2():
    # Testing problem 1 with different time steps. In particular, it verifies that when changing the time step the consumption of the heat pump remains approximately constant
    nodes = [esc.ElectricalNode("contatore")]
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
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
    env = esc.Environment(nodes=nodes, components=components, controllers = controllers, sensors=sensors)  # dt = 60 s
    for time_step in [1, 0.5, 0.25, 1/6, 5/60, 1/60]:
    # Test that results remain similar when changing the time step
        env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
        df_nodes, df_comps = env.to_dataframe()
        heat_pump_energy_demand = -df_comps[('heat_pump', 'contatore')].sum() * time_step
        assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)

def test_3():
    # Trying a more complex system, with PV panels 
    nodes = [esc.ElectricalNode("contatore")]
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        esc.PVpanelFromPVGIS(name = 'pv_panels', electrical_node='contatore', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        esc. ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10)
    ]
    sensors = [
        esc.TemperatureSensor('storage_tank_temperature_sensor', node_name = 'hot_water_storage_thermal_node')
    ]

    env = esc.Environment(nodes=nodes, components=components, controllers = controllers, sensors = sensors)  # dt = 60 s
    time_step = 0.5
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    heat_pump_energy_demand = -df_comps[('heat_pump', 'contatore')].sum() * time_step
    assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    electricity_from_pv = df_comps[('pv_panels', 'contatore')].sum() * time_step
    assert math.isclose(electricity_from_pv, 27, abs_tol = 2)
    electricity_demand = df_comps[('electric_grid', 'contatore')].sum() * time_step
    electricity_from_grid = df_comps.loc[df_comps[('electric_grid', 'contatore')] > 0, ('electric_grid', 'contatore')].sum() * time_step
    electricity_to_grid = -df_comps.loc[df_comps[('electric_grid', 'contatore')] < 0, ('electric_grid', 'contatore')].sum() * time_step
    assert math.isclose(electricity_demand, -15, abs_tol = 2)
    assert math.isclose(electricity_from_grid, 8, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 24, abs_tol = 2)
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 325, abs_tol = 1)

def test_4():
    # Like test 3, but adding a battery with related controller
    # Trying a more complex system, with PV panels
    nodes = [esc.ElectricalNode("inverter")]
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'inverter', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        esc.PVpanelFromPVGIS(name = 'pv_panels', electrical_node='inverter', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.LithiumIonBattery(name = 'battery', electrical_node='inverter', capacity = 2.0, SOC_0 = 0.5),
        esc.BalancingUtility(name = 'electric_grid', nodes = ['inverter']),
        esc.ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        esc.Inverter('inverter', 'battery', 'battery_SOC_sensor', 'grid_exchange_power_sensor')
    ]
    sensors = [
        esc.TemperatureSensor('storage_tank_temperature_sensor', node_name = 'hot_water_storage_thermal_node'),
        esc.SOCSensor('battery_SOC_sensor', 'battery', 'battery_electrical_node'),
        esc.PowerBalanceSensor('grid_exchange_power_sensor', 'inverter')
    ]

    env = esc.Environment(nodes=nodes, components=components, controllers = controllers, sensors = sensors)  # dt = 60 s
    time_step = 0.5
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    heat_pump_energy_demand = -df_comps[('heat_pump', 'inverter')].sum() * time_step
    assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    electricity_from_pv = df_comps[('pv_panels', 'inverter')].sum() * time_step
    assert math.isclose(electricity_from_pv, 27, abs_tol = 2)
    electricity_demand = df_comps[('electric_grid', 'inverter')].sum() * time_step
    electricity_from_grid = df_comps.loc[df_comps[('electric_grid', 'inverter')] > 0, ('electric_grid', 'inverter')].sum() * time_step
    electricity_to_grid = -df_comps.loc[df_comps[('electric_grid', 'inverter')] < 0, ('electric_grid', 'inverter')].sum() * time_step
    assert math.isclose(electricity_demand, -15, abs_tol = 2)
    assert math.isclose(electricity_from_grid, 0, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 15, abs_tol = 2)
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 325, abs_tol = 1)