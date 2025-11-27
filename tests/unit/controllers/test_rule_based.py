import energy_system_control as esc
import pytest, math, os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def test_base_controller():
    # This test is simply test 4 of the overall examples.
    # In a first run, we use the "standard" controller, while in a second run we test what happens with a more optimised, rule-based controller
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_max = 1.5, COP = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=2.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
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
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.ElectricPowerSensor('PV_power_sensor', 'inverter_PV_input_port')
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
    env_base = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    time_step = 0.25
    env_base.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_ports, df_controllers, df_sensors = env_base.to_dataframe()
    df_ports.to_csv(os.path.join(__TEST__, 'PLAYGROUND', 'test_rule_based_1_results_ports.csv'), sep = ";")
    df_sensors.to_csv(os.path.join(__TEST__, 'PLAYGROUND', 'test_rule_based_1_results_sensors.csv'), sep = ";")
    electricity_to_grid_base = df_ports.loc[df_ports['electric_grid_electricity_port:electricity'] > 0, 'electric_grid_electricity_port:electricity'].sum() / 3600
    electricity_from_grid_base = -df_ports.loc[df_ports['electric_grid_electricity_port:electricity'] < 0, 'electric_grid_electricity_port:electricity'].sum() / 3600
    assert math.isclose(electricity_to_grid_base, 6, abs_tol = 2)
    assert math.isclose(electricity_from_grid_base, 3, abs_tol = 2)

    controllers = [
        esc.HeatPumpRuleBasedController('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 'PV_power_sensor', 40, 10, 0.600),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
    ]
    env_opti = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    time_step = 0.25
    env_opti.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_ports, df_controllers, df_sensors = env_opti.to_dataframe()
    df_ports.to_csv(os.path.join(__TEST__, 'PLAYGROUND', 'test_rule_based_2_results_ports.csv'), sep = ";")
    df_sensors.to_csv(os.path.join(__TEST__, 'PLAYGROUND', 'test_rule_based_2_results_sensors.csv'), sep = ";")
    electricity_to_grid_opti = df_ports.loc[df_ports['electric_grid_electricity_port:electricity'] > 0, 'electric_grid_electricity_port:electricity'].sum() / 3600
    electricity_from_grid_opti = -df_ports.loc[df_ports['electric_grid_electricity_port:electricity'] < 0, 'electric_grid_electricity_port:electricity'].sum() / 3600
    assert electricity_to_grid_opti < electricity_to_grid_base
    assert electricity_from_grid_opti < electricity_from_grid_base
