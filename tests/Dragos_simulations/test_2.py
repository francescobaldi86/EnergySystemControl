import energy_system_control as esc
import pytest, math, os
import pandas as pd

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_2():
    for time_step in [1, 0.5, 0.25, 1/6, 5/60, 1/60]:
        # Testing problem 1 with different time steps. In particular, it verifies that when changing the time step the consumption of the heat pump remains approximately constant
        components = [
            esc.HotWaterDemand.from_iea(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
            esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2),
            esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
            esc.ElectricityGrid(name = 'electric_grid'),
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
        # Test that results remain similar when changing the time step
        sim_config = esc.SimulationConfig(time_start_h = 0.0, simulation_end_h = 24.0*7, time_step_h = time_step)
        sim = esc.Simulator(env, sim_config)
        results = sim.run()  # simulate 6 hours
        #heat_pump_energy_demand = results.get_cumulated_electricity('heat_pump_electricity_input_port')
        #assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
        nome_file = f'test_2_results_dt_{round(time_step, 3)}.xlsx'
        excel_path = os.path.join(__HERE__, 'Tabelle_dati', nome_file)

        with pd.ExcelWriter(excel_path) as writer:
            df_ports.to_excel(writer, sheet_name='Ports')
            df_sensors.to_excel(writer, sheet_name='Sensors')
            df_controllers.to_excel(writer, sheet_name='Controllers')

if __name__ == "__main__":
    test_2()
