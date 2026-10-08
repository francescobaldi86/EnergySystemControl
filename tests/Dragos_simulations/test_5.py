import energy_system_control as esc
import pytest, math, os
import pandas as pd

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_5():
    # Like test 3, but adding a battery with related controller
    components = [
        esc.HotWaterDemand.from_iea(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.ElectricityGrid(name = 'electric_grid'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'electricity_demand_sensor', 'pv_power_sensor')
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.ElectricPowerSensor('pv_power_sensor', 'inverter_PV_input_port'),
        esc.ElectricPowerSensor('electricity_demand_sensor', 'inverter_AC_output_port_0'),
        esc.ElectricPowerSensor('battery_power_sensor', 'battery_electricity_port'),
        esc.ElectricPowerSensor('grid_power_sensor', 'electric_grid_electricity_port')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_AC_output_port_0'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port')
    ]
    # Create environment
    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    # Create simulator object
    sim_config = esc.SimulationConfig(time_start_h = 0.0, simulation_end_h = 24.0*7, time_step_h = 1/60)
    sim = esc.Simulator(env, sim_config)
    # Run simulation
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()

    os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', "test_5_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
            df_ports.to_excel(writer, sheet_name='Ports')
            df_sensors.to_excel(writer, sheet_name='Sensors')
            df_controllers.to_excel(writer, sheet_name='Controllers')
    
if __name__ == "__main__":
    test_5()