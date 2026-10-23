import energy_system_control as esc
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__PLAYGROUND__ = os.path.join(__TEST__, 'PLAYGROUND', 'Dragos')

def test_7():
    components = [
        esc.ElectricityGrid(name='electric_grid'),
        esc.AnaerobicDigester(name='biogas_producer', biogas_rate=15, methane_fraction=0.65),
        esc.BiogasStorage(name='biogas_storage', capacity=500.0, initial_ch4_mass=50.0, initial_co2_mass=60.0),
        esc.InternalCombustionEngine(name='engine', P_el_design=50.0, efficiency_el=0.38, fuel='biogas'),
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
                              storage_sensor='biogas_SOC_sensor',
                              biogas_soc_on_threshold=0.8,
                              biogas_soc_off_threshold=0.2)
    ]
    
    connections = [
        # Circuito Fluido (Biogas)
        ('biogas_storage_input', 'biogas_producer_biogas_port'), 
        ('engine_biogas_port', 'biogas_storage_output'),
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

    os.makedirs(__PLAYGROUND__, exist_ok=True)
    excel_path = os.path.join(__PLAYGROUND__, "test_7_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_ports.to_excel(writer, sheet_name='Ports')
        df_sensors.to_excel(writer, sheet_name='Sensors')
        df_controllers.to_excel(writer, sheet_name='Controllers')

if __name__ == "__main__":
    test_7()