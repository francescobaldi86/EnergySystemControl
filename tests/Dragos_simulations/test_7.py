import energy_system_control as esc
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_7():
    components = [
        esc.ElectricityGrid(name='electric_grid'),
        esc.AnaerobicDigester(name='biogas_producer', biogas_rate=15, methane_fraction=0.65),
        esc.BiogasStorage(name='biogas_storage', capacity=500.0, initial_ch4_mass=50.0, initial_co2_mass=60.0),
        esc.InternalCombustionEngine(name='engine', P_el_design=50.0, efficiency_el=0.38),
        esc.ElectricityDemand.from_csv(name='demand', 
                                       path=os.path.join(__TEST__, 'DATA', 'yearly_data_electricity_demand_15min.csv'), 
                                       var_unit='kWh',
                                       rescale_factor = 1000, 
                                       column_name='value', 
                                       time_alignment='yearly'),
        esc.Bus(name='ac_bus', ports_info={
                    'ac_bus_engine_port': 'electricity',
                    'ac_bus_grid_port': 'electricity',
                    'ac_bus_demand_port': 'electricity'
                })
    ]
    
    sensors = [
        esc.SOCSensor('biogas_SOC_sensor', 'biogas_storage'),
        esc.ElectricPowerSensor('electricity_demand_sensor', 'demand_electricity_port'),
        esc.ElectricPowerSensor('engine_power_sensor', 'engine_electricity_port'),
        esc.ElectricPowerSensor('grid_exchange_sensor', 'electric_grid_electricity_port')
    ]
    
    controllers = [
        esc.EngineOnOffController('engine_controller', 
                              engine_name='engine', 
                              demand_sensor='electricity_demand_sensor', 
                              storage_sensor='biogas_SOC_sensor')
    ]
    
    connections = [
        # Circuito Fluido (Biogas)
        ('biogas_storage_input', 'biogas_producer_fluid_port'), 
        ('engine_fuel_port', 'biogas_storage_output'),
        # Circuito Elettrico: Motore, Rete e Domanda convergono tutti sulle porte del Bus
        ('engine_electricity_port', 'ac_bus_engine_port'),
        ('electric_grid_electricity_port', 'ac_bus_grid_port'),
        ('demand_electricity_port', 'ac_bus_demand_port')
    ]
    
    # 5. Creazione dell'ambiente e configurazione della simulazione
    env = esc.Environment(components=components, controllers=controllers, sensors=sensors, connections=connections)
    sim_config = esc.SimulationConfig(time_start_h=0.0, simulation_end_h=24.0*7, time_step_h=1/60)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()

    os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', "test_7_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_ports.to_excel(writer, sheet_name='Ports')
        df_sensors.to_excel(writer, sheet_name='Sensors')
        df_controllers.to_excel(writer, sheet_name='Controllers')

    start_step = 6 * 1440
    end_step = 7 * 1440
    
    engine_power = abs(df_ports['engine_electricity_port:electricity'].iloc[start_step:end_step])
    demand = abs(df_ports['demand_electricity_port:electricity'].iloc[start_step:end_step])
    grid_exchange = df_ports['electric_grid_electricity_port:electricity'].iloc[start_step:end_step]
    
    biogas_soc = df_sensors['biogas_SOC_sensor'].iloc[start_step:end_step]
    biogas_fuel_mass = abs(df_ports['engine_fuel_port:mass'].iloc[start_step:end_step])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Bilancio Elettrico
    ax1.plot(engine_power.index, engine_power, color='darkorange', label='Generazione Motore', linewidth=2)
    ax1.plot(demand.index, demand, color='black', label='Domanda Elettrica', linewidth=2, linestyle='--')
    # Il flusso entra nella rete (positivo) -> Export
    ax1.fill_between(grid_exchange.index, 0, grid_exchange.clip(lower=0), color='green', alpha=0.3, label='Export verso Rete')
    # Il flusso esce dalla rete (negativo) -> Import
    ax1.fill_between(grid_exchange.index, 0, grid_exchange.clip(upper=0), color='red', alpha=0.3, label='Import dalla Rete')
    
    ax1.set_ylabel('Potenza (kW)')
    ax1.set_title('Test 7: Bilancio Elettrico e Consumo Biogas (Prime 48 ore)')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot 2: Stato del Serbatoio Biogas e Consumo Motore
    ax2.plot(biogas_fuel_mass.index, biogas_fuel_mass, color='purple', label='Flusso Massa Combustibile (kg/s)')
    
    ax2_soc = ax2.twinx()
    ax2_soc.plot(biogas_soc.index, biogas_soc, color='blue', label='Stato di Carica Biogas (SOC)')
    ax2_soc.set_ylabel('SOC (-)', color='blue')
    ax2_soc.set_ylim(0, 1.05)
    
    ax2.set_xlabel('Tempo (ore)')
    ax2.set_ylabel('Portata Massa (kg/s)')
    ax2.legend(loc='upper left')
    ax2_soc.legend(loc='upper right')
    ax2.grid(True)

    plot_path = os.path.join(__HERE__, 'Tabelle_dati', 'test_7_cogen_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    test_7()