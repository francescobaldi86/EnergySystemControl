import energy_system_control as esc
import pytest, math, os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def test_opti_controller():
    # This test is simply test 4 of the overall examples.
    # In a first run, we use the "standard" controller, while in a second run we test what happens with a more optimised, rule-based controller
    components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpLorentzEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2, heat_capacity_loss = 0.01),
        esc.MultiNodeHotWaterTank(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        esc.BalancingUtility(name = 'electric_grid', utility_type = 'electricity'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=1.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.Inverter(name = 'inverter'),
        esc.LithiumIonBattery(name = 'battery', capacity = 0.5, SOC_0 = 0.5),
    ]
    controllers = [
        esc.HeatPumpRuleBasedController('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 'PV_power_sensor', 40, 10, 0.600),
        esc.InverterController('inverter_controller', 'inverter', 'battery')
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.ElectricPowerSensor('PV_power_sensor', 'inverter_PV_input_port'),
        esc.SOCSensor('battery_SOC_sensor', 'battery')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_output_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port'),
    ]
    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    time_step = 1/60  # One minute time step
    time_end = 24.0*7  # One week simulation
    sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = time_end, time_step_h = time_step)
    for control_pv_setpoint in [4.0, 0.4]:
        env.controllers['heat_pump_controller'].power_PV_activation = control_pv_setpoint
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        electricity_to_grid_base = results.get_cumulated_electricity('electric_grid_electricity_port', unit = "kWh", sign = "only positive")
        electricity_from_grid_base = results.get_cumulated_electricity('electric_grid_electricity_port', unit = "kWh", sign = "only negative")
    assert True
