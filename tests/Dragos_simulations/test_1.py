import energy_system_control as esc
import pytest, math, os
import pandas as pd
import matplotlib.pyplot as plt

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_1():
    # Represent substantially a minimal example of electric heating system
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
    time_step = 1/60
    sim_config = esc.SimulationConfig(time_start_h = 0.0, simulation_end_h = 24.0*7, time_step_h = time_step)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    #assert math.isclose(df_sensors.loc[10.0, 'storage_tank_temperature_sensor'], 323, abs_tol = 1)
    #df_ports.to_csv(os.path.join(__TEST__, 'PLAYGROUND', 'test_1_results_ports.csv'), sep = ";")
    os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', 'test_1_results.xlsx')

    with pd.ExcelWriter(excel_path) as writer:
        df_ports.to_excel(writer, sheet_name='Ports')
        df_sensors.to_excel(writer, sheet_name='Sensors')
        df_controllers.to_excel(writer, sheet_name='Controllers')

    fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Asse Y Primario: Potenze Termiche
    color_demand = 'tab:red'
    color_hp = 'tab:orange'
    ax1.set_xlabel('Tempo (ore)')
    ax1.set_ylabel('Potenza Termica (kW)', color='black')
        
        # Plottiamo la potenza richiesta dall'utenza e la potenza fornita dalla PdC
    ax1.plot(df_ports.index, df_ports['demand_DHW_fluid_port:heat'], color=color_demand, label='Prelievo ACS (Demand)', alpha=0.7)
    ax1.plot(df_ports.index, df_ports['heat_pump_heat_output_port:heat'], color=color_hp, label='Erogazione Pompa di Calore', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

        # Asse Y Secondario: Temperatura dell'Accumulo
    ax2 = ax1.twinx()  
    color_temp = 'tab:blue'
    ax2.set_ylabel('Temperatura Accumulo (°C o K)', color=color_temp)  
        
        # Assumiamo che il sensore dell'accumulo sia nei risultati df_sensors
    ax2.plot(df_sensors.index, df_sensors['storage_tank_temperature_sensor'], color=color_temp, label='Temp. Accumulo', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_temp)
    ax2.legend(loc='upper right')

    plt.title('Test 1: Bilancio Termico e Inerzia dell\'Accumulo')
    plt.grid(True, linestyle='--', alpha=0.5)
        
    plot_path = os.path.join(__HERE__, 'Tabelle_dati', 'test_1_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    test_1()