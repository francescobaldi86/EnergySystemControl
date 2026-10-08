import energy_system_control as esc
import pytest, math, os
import pandas as pd

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_3():
    # Trying a more complex system, with PV panels
    components = [
        esc.HotWaterDemand.from_iea(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.ElectricityGrid(name = 'electric_grid'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.Inverter(name = 'inverter')
    ]

    for comp in components:
        if comp.name == 'inverter':
            print("Porte dell'inverter disponibili:", comp.ports.keys())

    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        # esc.InverterController('inverter_controller', 'inverter', 'pv_power_sensor', 'electricity_demand_sensor')
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.ElectricPowerSensor('pv_power_sensor', 'inverter_PV_input_port'),
        esc.ElectricPowerSensor('electricity_demand_sensor', 'inverter_AC_output_port_0')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_AC_output_port_0'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port')
    ]
    # Create environment
    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    # Create simulator object
    sim_config = esc.SimulationConfig(time_start_h = 0.0, simulation_end_h = 24.0*7, time_step_h = 0.5)
    sim = esc.Simulator(env, sim_config)
    # Run simulation
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()

    os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', "test_3_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
            df_ports.to_excel(writer, sheet_name='Ports')
            df_sensors.to_excel(writer, sheet_name='Sensors')
            df_controllers.to_excel(writer, sheet_name='Controllers')

    # Verify results
    #heat_pump_energy_demand = results.get_cumulated_electricity('heat_pump_electricity_input_port')
    #electricity_from_pv = results.get_cumulated_electricity('inverter_PV_input_port')
    #net_electricity_demand = results.get_cumulated_electricity('electric_grid_electricity_port')
    #electricity_to_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only positive')
    #electricity_from_grid = results.get_cumulated_electricity('electric_grid_electricity_port', sign='only negative')

    #assert math.isclose(electricity_from_pv, 27, abs_tol = 2)
    #assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    #assert math.isclose(electricity_from_grid, 9, abs_tol = 2)
    #assert math.isclose(electricity_to_grid, 22, abs_tol = 2)
    #assert math.isclose(net_electricity_demand, 13, abs_tol = 2)
    #assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 325, abs_tol = 1)

if __name__ == "__main__":
    test_3()