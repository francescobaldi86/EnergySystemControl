import energy_system_control as esc
import pytest, math, os
import pandas as pd

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

'''def test_6():
    # Like test 5, but adding an ICE with related controller and sensors
    components = [
        esc.HotWaterDemand.from_iea(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        esc.ElectricityGrid(name = 'electric_grid'),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter'),
        esc.InternalCombustionEngine(name='backup_engine', P_el_design=2.5, efficiency_el=0.33),
        esc.GasGrid(name='gas_grid', utility_type='fluid')
    ]
    controllers = [
        esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
        esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'electricity_demand_sensor', 'pv_power_sensor'),
        esc.EngineRuleBasedController(name='engine_controller',
                                      controlled_component='backup_engine',
                                      electricity_demand_sensor='electricity_demand_sensor', 
                                      pv_power_sensor='pv_power_sensor',
                                      battery_soc_sensor = 'battery_SOC_sensor',
                                      net_power_activation=0.2, 
                                      min_time_on_h=0.75,
                                      min_soc_activation=0.7)
    ]
    sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.ElectricPowerSensor('pv_power_sensor', 'inverter_PV_input_port'),
        esc.ElectricPowerSensor('electricity_demand_sensor', 'inverter_AC_output_port_0'),
        esc.ElectricPowerSensor('battery_power_sensor', 'battery_electricity_port'),
        esc.ElectricPowerSensor('grid_power_sensor', 'electric_grid_electricity_port'),
        esc.ElectricPowerSensor('engine_power_sensor', 'backup_engine_electricity_port'),
        esc.PowerSensor('engine_fuel_sensor', 'backup_engine_fuel_port', flow_type='mass')
    ]
    connections = [
        ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
        ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
        ('heat_pump_electricity_input_port', 'inverter_AC_output_port_0'),
        ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ('inverter_PV_input_port', 'pv_panels_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port'),
        ('backup_engine_electricity_port', 'inverter_engine_port'),
        ('gas_grid_fluid_port', 'backup_engine_fuel_port')
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
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', "test_6_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
            df_ports.to_excel(writer, sheet_name='Ports')
            df_sensors.to_excel(writer, sheet_name='Sensors')
            df_controllers.to_excel(writer, sheet_name='Controllers')
    
if __name__ == "__main__":
    test_6()
'''

def test_6(mode='unified'):
    # Like test 5, but adding an ICE with related controller and sensors
    # Si aggiungono i controllori ChargeControllerWithEngine (relativo al controllo contemporaneo) e BatteryControlAware

    components = [
            esc.HotWaterDemand.from_iea(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
            esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design = 3.2),
            esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
            esc.ElectricityGrid(name = 'electric_grid'),
            esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
            esc.PVpanelFromPVGIS(name = 'pv_panels', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
            esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
            esc.Inverter(name = 'inverter'),
            esc.InternalCombustionEngine(name='backup_engine', P_el_design=2.5, efficiency_el=0.33),
            esc.GasGrid(name='gas_grid', utility_type='fluid')
        ]

    if mode == "unified":
        controllers = [
            esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
            esc.ChargeControllerWithEngine(
                name='unified_controller',
                battery_name='battery',
                engine_name='backup_engine',
                battery_soc_sensor='battery_SOC_sensor',
                electricity_demand_sensor='electricity_demand_sensor',
                pv_power_sensor='pv_power_sensor',
                net_power_activation=0.2,
                engine_p_el_design=2.5, # Corrisponde al P_el_design in InternalCombustionEngine
                min_soc_activation=0.7,
                min_time_on_h=0.75
            )
        ]
    elif mode == "separated":
        controllers = [
            esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'storage_tank_temperature_sensor', 40, 10),
            esc.BatteryControllerAware(
                name='charge_controller_aware',
                controlled_component='battery',
                soc_sensor='battery_SOC_sensor',
                demand_sensor='electricity_demand_sensor',
                pv_sensor='pv_power_sensor',
                engine_sensor='engine_power_sensor' # Sfrutta il sensore del motore
            ),
            esc.EngineRuleBasedController(
                name='engine_controller',
                controlled_component='backup_engine',
                electricity_demand_sensor='electricity_demand_sensor', 
                pv_power_sensor='pv_power_sensor',
                battery_soc_sensor='battery_SOC_sensor',
                net_power_activation=0.2, 
                min_time_on_h=0.75,
                min_soc_activation=0.7
            )
        ]
    else:
        raise ValueError("Il parametro 'mode' deve essere 'unified' o 'separated'.")

    sensors = [
            esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
            esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
            esc.SOCSensor('battery_SOC_sensor', 'battery'),
            esc.ElectricPowerSensor('pv_power_sensor', 'inverter_PV_input_port'),
            esc.ElectricPowerSensor('electricity_demand_sensor', 'inverter_AC_output_port_0'),
            esc.ElectricPowerSensor('battery_power_sensor', 'battery_electricity_port'),
            esc.ElectricPowerSensor('grid_power_sensor', 'electric_grid_electricity_port'),
            esc.ElectricPowerSensor('engine_power_sensor', 'backup_engine_electricity_port'),
            esc.PowerSensor('engine_fuel_sensor', 'backup_engine_fuel_port', flow_type='mass')
        ]
    connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port_0'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port'),
            ('backup_engine_electricity_port', 'inverter_engine_port'),
            ('gas_grid_fluid_port', 'backup_engine_fuel_port')
        ]

    env = esc.Environment(components=components, controllers = controllers, sensors=sensors, connections=connections)  # dt = 60 s
    sim_config = esc.SimulationConfig(time_start_h = 0.0, simulation_end_h = 24.0*7, time_step_h = 1/60)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()

    os.makedirs(os.path.join(__HERE__, 'Tabelle_dati'), exist_ok=True)
    excel_path = os.path.join(__HERE__, 'Tabelle_dati', f"test_6_results_v2_{mode}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_ports.to_excel(writer, sheet_name='Ports')
        df_sensors.to_excel(writer, sheet_name='Sensors')
        df_controllers.to_excel(writer, sheet_name='Controllers')
    print(f"Simulazione completata con mode='{mode}'. Dati salvati in: {excel_path}")

if __name__ == "__main__":
    # si eseguono entrambe le simulazioni in sequenza per confrontare le tabelle dati
    test_6(mode="separated")
    test_6(mode="unified")