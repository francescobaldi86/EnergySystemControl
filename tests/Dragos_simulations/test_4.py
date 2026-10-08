import energy_system_control as esc
import pytest, math, os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_4():
    # Represents a minimal test of an electric system with PV and a battery, that requires control
    components = [
        esc.ElectricityGrid(name = 'electric_grid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=0.8, latitude=44.500365, longitude=11.350096, tilt=90, azimuth=90),
        esc.LithiumIonBattery(name = 'battery', capacity = 1.5, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter'),
        esc.ElectricityDemand.from_csv(name = 'demand', path = os.path.join(__TEST__, 'DATA', 'yearly_data_electricity_demand_15min.csv'), var_unit = 'kWh', column_name = 'value', time_alignment = 'yearly')
    ]
    controllers = [
        esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'electricity_demand_sensor', 'pv_power_sensor')
    ]
    sensors = [
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.ElectricPowerSensor('pv_power_sensor', 'inverter_PV_input_port'),
        esc.ElectricPowerSensor('electricity_demand_sensor', 'inverter_AC_output_port_0'),
        esc.ElectricPowerSensor('grid_exchange_sensor', 'electric_grid_electricity_port'),
        esc.ElectricPowerSensor('battery_power_sensor', 'battery_electricity_port'),
    ]
    connections = [
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port'),
        ('inverter_AC_output_port_0', 'demand_electricity_port')
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
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', "test_4_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
            df_ports.to_excel(writer, sheet_name='Ports')
            df_sensors.to_excel(writer, sheet_name='Sensors')
            df_controllers.to_excel(writer, sheet_name='Controllers')

    start_step = 6*1440
    end_step = 7*1440
    
    pv_power = abs(df_ports['pv_panels_electricity_port:electricity'].iloc[start_step:end_step])
    demand = abs(df_ports['demand_electricity_port:electricity'].iloc[start_step:end_step])
    
    # lo scambio verso la rete è positivo, il prelievo negativo 
    grid_exchange = df_ports['electric_grid_electricity_port:electricity'].iloc[start_step:end_step]
    battery = df_ports['battery_electricity_port:electricity'].iloc[start_step:end_step]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot del confronto generazione vs domanda vs rete
    ax1.plot(pv_power.index, pv_power, color='gold', label='Generazione PV', linewidth=2)
    ax1.plot(demand.index, demand, color='black', label='Domanda Elettrica', linewidth=2, linestyle='--')
    ax1.fill_between(grid_exchange.index, 0, grid_exchange.clip(lower=0), color='red', alpha=0.3, label='Import dalla Rete')
    ax1.fill_between(grid_exchange.index, 0, grid_exchange.clip(upper=0), color='green', alpha=0.3, label='Export verso Rete')
    
    ax1.set_ylabel('Potenza (kW)')
    ax1.set_title('Test 4: Bilancio Elettrico Microgrid (Prime 48 ore)')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot dello stato della batteria
    ax2.plot(battery.index, battery, color='purple', label='Potenza Batteria (>0 Scarica, <0 Carica)')
    
    # Stato di carica (SOC) sull'asse Y destro
    ax2_soc = ax2.twinx()
    soc = df_sensors['battery_SOC_sensor'].iloc[start_step:end_step]
    ax2_soc.plot(soc.index, soc, color='blue', label='Stato di Carica (SOC)')
    ax2_soc.set_ylabel('SOC (%)', color='blue')
    ax2_soc.set_ylim(0, 1.05)
    
    ax2.set_xlabel('Tempo (ore)')
    ax2.set_ylabel('Potenza Batteria (kW)')
    ax2.legend(loc='upper left')
    ax2_soc.legend(loc='upper right')
    ax2.grid(True)

    plot_path = os.path.join(__HERE__, 'Tabelle_dati', 'test_4_microgrid_plot_7.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    test_4()