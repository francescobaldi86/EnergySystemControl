import energy_system_control as esc
import os
from energy_system_control import components
from energy_system_control import components
import pandas as pd
from datetime import datetime

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def test_8():
    # Similar to test_6, but with the biogas producer fueling the biogas storage connected to the internal combustion engine.
    components = [
        esc.HotWaterDemand.from_iea(name="demand_DHW", reference_temperature=40, profile_name='L'),
        esc.HeatPumpConstantEfficiency(name='heat_pump', Qdot_design=1.5, COP_design=3.2),
        esc.HotWaterStorage(name='hot_water_storage', max_temperature=80, tank_volume=200, T_0=45, convection_coefficient_losses=0.0),
        esc.ElectricityGrid(name='electric_grid'),
        esc.ColdWaterGrid(name='water_grid', utility_type='fluid'),
        esc.PVpanelFromData(name='pv_panels_est', data_path=os.path.join(__TEST__, 'DATA'), filename='pvgis_data.csv', column_name='P', skipfooter=11, rescale_factor=3.6, time_alignment='yearly'),
        esc.PVpanelFromData(name='pv_panels_ovest', data_path=os.path.join(__TEST__, 'DATA'), filename='pvgis_data.csv', column_name='P', skipfooter=11, rescale_factor=3.6, time_alignment='yearly'),
        esc.LithiumIonBattery(name='battery', capacity=2.0, SOC_0=0.5),
        esc.Inverter(name='inverter', number_of_ac_output_ports=2, number_of_pv_input_ports=2),
        esc.AnaerobicDigester(name='biogas_producer', biogas_rate=15, methane_fraction=0.65),
        esc.BiogasStorage(name='biogas_storage', capacity=500.0, initial_ch4_mass=50.0, initial_co2_mass=60.0),
        esc.InternalCombustionEngine(name='backup_engine', P_el_design=30, efficiency_el=0.33, fuel='biogas'),
        esc.ElectricityDemand.from_csv(name='demand_rifugio', path=os.path.join(__TEST__, 'DATA', 'consumi_totali_rifugio_2026.csv'),
                                     var_unit='kWh', rescale_factor=0.001, column_name='Power', time_alignment='yearly', date_format='%d/%m/%Y %H:%M')
    ]

    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        esc.BiogasEngineRuleBasedController(
            name='biogas_engine_rule_based_controller',
            battery_name='battery',
            engine_name='backup_engine',
            battery_soc_sensor='battery_SOC_sensor',
            biogas_soc_sensor='biogas_SOC_sensor',
            building_demand_sensor='building_demand_sensor',  # Dati CSV
            hp_demand_sensor='hp_demand_sensor',              # Pompa Calore
            pv_power_sensor_est='pv_power_sensor_est',
            pv_power_sensor_ovest='pv_power_sensor_ovest',
            net_power_activation=0.2,
            engine_p_el_design=30,
            min_battery_soc_activation=0.7,
            biogas_soc_on_threshold=0.8,
            biogas_soc_off_threshold=0.2,
            min_time_on_h=0.75
        )
    ]

    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.SOCSensor('biogas_SOC_sensor', 'biogas_storage'),
        esc.ElectricPowerSensor('pv_power_sensor_est', 'inverter_PV_input_port_0'),
        esc.ElectricPowerSensor('pv_power_sensor_ovest', 'inverter_PV_input_port_1'),
        
        esc.ElectricPowerSensor('hp_demand_sensor', 'inverter_AC_output_port_0'), 
        esc.ElectricPowerSensor('building_demand_sensor', 'inverter_AC_output_port_1'),
        
        esc.ElectricPowerSensor('battery_power_sensor', 'battery_electricity_port'),
        esc.ElectricPowerSensor('grid_power_sensor', 'electric_grid_electricity_port'),
        esc.ElectricPowerSensor('engine_power_sensor', 'backup_engine_electricity_port'),
        esc.PowerSensor('engine_fuel_sensor', 'backup_engine_biogas_port', flow_type='mass') 
    ]

    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port_0', 'pv_panels_est_electricity_port'),
        ('inverter_PV_input_port_1', 'pv_panels_ovest_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port'),
        ('backup_engine_electricity_port', 'inverter_engine_port'),
        ('biogas_storage_input', 'biogas_producer_biogas_port'), 
        ('backup_engine_biogas_port', 'biogas_storage_output'),
        ('heat_pump_electricity_input_port', 'inverter_AC_output_port_0'),
        ('demand_rifugio_electricity_port', 'inverter_AC_output_port_1')
    ]

    env = esc.Environment(components=components, controllers=controllers, sensors=sensors, connections=connections)
    sim_config = esc.SimulationConfig(time_start_h=0.0, simulation_end_h=1441.0, time_step_h=1/60, simulation_start_datetime=datetime(2026, 7, 25, 13, 0))
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    
    df_ports, df_controllers, df_sensors = results.to_dataframe()

    os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', "test_8_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_ports.to_excel(writer, sheet_name='Ports')
        df_sensors.to_excel(writer, sheet_name='Sensors')
        df_controllers.to_excel(writer, sheet_name='Controllers')
    
    print(f"Simulazione test_8 completata. Dati salvati in: {excel_path}")

if __name__ == "__main__":
    test_8()