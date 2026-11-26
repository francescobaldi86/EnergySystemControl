import energy_system_control as esc
import os
import pandas as pd
from datetime import datetime

# Classi introdotte per test_9 (file nuovi: il resto della libreria non e' stato modificato,
# a parte la correzione dei tempi minimi in controllers/base.py e dell'autoscarica in electric_storage.py)
from energy_system_control.components.controlled_components.dual_fuel_engine import DualFuelEngine
from energy_system_control.components.grids.fuel_grids import LPGSupply, AtmosphericVent
from energy_system_control.components.storage_units.biogas_storage_with_vent import BiogasStorageWithVent
from energy_system_control.components.composite_components.lead_acid_battery_corrected import LeadAcidBatteryCorrected
from energy_system_control.controllers.rule_based_dual_fuel import DualFuelEngineRuleBasedController

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__PLAYGROUND__ = os.path.join(__TEST__, 'PLAYGROUND', 'Dragos')

GWP_CH4 = 27.0   # GWP a 100 anni del metano non fossile (IPCC AR6), per la CO2 equivalente dello sfiato


def test_9():
    # Rispetto a test_8, configurazione piu' vicina al rifugio:
    # - utenze: consumi elettrici del rifugio + cucina a induzione + caldaia ACS elettrica on/off
    #   (cucina e caldaia elettrificate a partire dai consumi di GPL della campagna 2026);
    # - eliminati pompa di calore, accumulo ACS da 200 l, profilo IEA e rete dell'acqua fredda;
    # - batteria al piombo BAE Secura PVV Solar 18 PVV 3420 (24 celle, 48 V, 125.28 kWh C10);
    # - digestore da 1 m3/h, gasometro da 20 m3 con sfiato a solo traboccamento (niente torcia);
    # - motore bi-fuel biogas/GPL da 30 kW; il surplus elettrico va in rete.
    components = [
        esc.ElectricityGrid(name='electric_grid'),
        esc.PVpanelFromPVGIS(name='pv_panels_est', installed_power=3.6, latitude=46.216, longitude=12.215, tilt=30, azimuth = -90, years=[2023]), #pannelli fotovoltaici orientati a est
        esc.PVpanelFromPVGIS(name='pv_panels_ovest', installed_power=3.6, latitude=46.216, longitude=12.215, tilt=30, azimuth = 90, years=[2023]), #pannelli fotovoltaici orientati a ovest
        LeadAcidBatteryCorrected(name='battery', capacity=125.28, SOC_0=0.5, SOC_min=0.3, SOC_derate=0.6, crate=0.2, erate=0.5, SOC_taper=0.8),
        esc.Inverter(name='inverter', number_of_ac_output_ports=3, number_of_pv_input_ports=2),
        esc.AnaerobicDigester(name='biogas_producer', biogas_rate=1.0, methane_fraction=0.65),
        BiogasStorageWithVent(name='biogas_storage', capacity=20.0, initial_soc=0.5, initial_methane_fraction=0.65),
        AtmosphericVent(name='biogas_vent'),
        DualFuelEngine(name='backup_engine', P_el_design=30, biogas_storage_name='biogas_storage', min_biogas_soc=0.2, polynomial_efficiency=True),
        LPGSupply(name='lpg_supply'),
        esc.ElectricityDemand.from_csv(name='demand_rifugio', path=os.path.join(__TEST__, 'DATA', 'consumi_totali_rifugio_2026.csv'),
                                     var_unit='kWh', rescale_factor=0.001, column_name='Power', time_alignment='yearly', date_format='%d/%m/%Y %H:%M'),
        esc.ElectricityDemand.from_csv(name='demand_cucina', path=os.path.join(__TEST__, 'DATA', 'consumi_cucina_induzione_2026.csv'),
                                     var_type='power', var_unit='W', rescale_factor=1.0, column_name='Power [W]', time_alignment='yearly', date_format='%d/%m/%Y %H:%M'),
        esc.ElectricityDemand.from_csv(name='demand_caldaia', path=os.path.join(__TEST__, 'DATA', 'consumi_caldaia_elettrica_2026.csv'),
                                     var_type='power', var_unit='W', rescale_factor=1.0, column_name='Power [W]', time_alignment='yearly', date_format='%d/%m/%Y %H:%M'),
    ]

    controllers = [
        DualFuelEngineRuleBasedController(
            name='dual_fuel_engine_controller',
            battery_name='battery',
            engine_name='backup_engine',
            biogas_soc_sensor='biogas_SOC_sensor',
            demand_sensors=['building_demand_sensor', 'kitchen_demand_sensor', 'boiler_demand_sensor'],
            pv_power_sensor_est='pv_power_sensor_est',
            pv_power_sensor_ovest='pv_power_sensor_ovest',
            engine_p_el_design=30,
            engine_load_factor=0.75,          # massimo rendimento della curva del motore
            min_engine_load_factor=0.15,
            biogas_soc_on_threshold=0.8,
            biogas_soc_off_threshold=0.2,     # uguale a min_biogas_soc del motore: sotto questa soglia si usa il GPL
            min_time_on_h=0.75,
            min_time_off_h=0.0
        ),
    ]

    sensors = [
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.SOCSensor('biogas_SOC_sensor', 'biogas_storage'),
        esc.ElectricPowerSensor('pv_power_sensor_est', 'inverter_PV_input_port_0'),
        esc.ElectricPowerSensor('pv_power_sensor_ovest', 'inverter_PV_input_port_1'),
        esc.ElectricPowerSensor('building_demand_sensor', 'inverter_AC_output_port_0'),  # Dati CSV rifugio
        esc.ElectricPowerSensor('kitchen_demand_sensor', 'inverter_AC_output_port_1'),   # Cucina a induzione
        esc.ElectricPowerSensor('boiler_demand_sensor', 'inverter_AC_output_port_2'),    # Caldaia ACS elettrica
        esc.ElectricPowerSensor('battery_power_sensor', 'battery_electricity_port'),
        esc.ElectricPowerSensor('grid_power_sensor', 'electric_grid_electricity_port'),
        esc.ElectricPowerSensor('engine_power_sensor', 'backup_engine_electricity_port'),
        esc.PowerSensor('engine_biogas_sensor', 'backup_engine_biogas_port', flow_type='mass'),
        esc.PowerSensor('engine_lpg_sensor', 'backup_engine_fuel_port', flow_type='mass'),
        esc.PowerSensor('vent_volume_sensor', 'biogas_vent_biogas_port', flow_type='volume', scale_factor=3600),
    ]

    connections = [
        ('inverter_PV_input_port_0', 'pv_panels_est_electricity_port'),
        ('inverter_PV_input_port_1', 'pv_panels_ovest_electricity_port'),
        ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ('inverter_ESS_port', 'battery_electricity_port'),
        ('backup_engine_electricity_port', 'inverter_engine_port'),
        ('biogas_storage_input', 'biogas_producer_biogas_port'),
        ('backup_engine_biogas_port', 'biogas_storage_output'),
        ('backup_engine_fuel_port', 'lpg_supply_fuel_port'),
        ('biogas_vent_biogas_port', 'biogas_storage_flare_output'),
        ('demand_rifugio_electricity_port', 'inverter_AC_output_port_0'),
        ('demand_cucina_electricity_port', 'inverter_AC_output_port_1'),
        ('demand_caldaia_electricity_port', 'inverter_AC_output_port_2'),
    ]

    env = esc.Environment(components=components, controllers=controllers, sensors=sensors, connections=connections)
    sim_config = esc.SimulationConfig(time_start_h=0.0, simulation_end_h=1441.0, time_step_h=1/60, simulation_start_datetime=datetime(2026, 7, 25, 13, 0))
    sim = esc.Simulator(env, sim_config)
    results = sim.run()

    df_ports, df_controllers, df_sensors = results.to_dataframe()

    # Indicatori principali (le porte sono in kW o kg/s: somma sui passi * durata del passo)
    dt_h = sim_config.time_step_h
    dt_s = dt_h * 3600
    ports = df_ports
    storage = env.components['biogas_storage']
    kpi = {
        'Biogas prodotto [m3]': -ports['biogas_producer_biogas_port:volume'].sum() * dt_s,
        'Biogas al motore [m3]': ports['backup_engine_biogas_port:volume'].sum() * dt_s,
        'Biogas sfiatato [m3]': ports['biogas_vent_biogas_port:volume'].sum() * dt_s,
        'Metano sfiatato [kg]': storage.vented_ch4_mass_total,
        'CO2 equivalente dello sfiato [kg]': storage.vented_ch4_mass_total * GWP_CH4,
        'GPL al motore [kg]': ports['backup_engine_fuel_port:mass'].sum() * dt_s,
        'Energia elettrica del motore [kWh]': -ports['backup_engine_electricity_port:electricity'].sum() * dt_h,
        'Ore di funzionamento del motore [h]': (df_controllers['dual_fuel_engine_controller:backup_engine'] > 0).sum() * dt_h,
        'Energia prelevata dalla rete [kWh]': ports['electric_grid_electricity_port:electricity'].clip(upper=0).sum() * -dt_h,
        'Energia ceduta alla rete [kWh]': ports['electric_grid_electricity_port:electricity'].clip(lower=0).sum() * dt_h,
        'SOC batteria iniziale/finale [-]': f"{df_sensors['battery_SOC_sensor'].iloc[0]:.3f} / {df_sensors['battery_SOC_sensor'].iloc[-1]:.3f}",
    }
    df_kpi = pd.DataFrame({'Valore': kpi})

    os.makedirs(__PLAYGROUND__, exist_ok=True)
    excel_path = os.path.join(__PLAYGROUND__, "test_9_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_ports.to_excel(writer, sheet_name='Ports')
        df_sensors.to_excel(writer, sheet_name='Sensors')
        df_controllers.to_excel(writer, sheet_name='Controllers')
        df_kpi.to_excel(writer, sheet_name='Indicatori')

    print(df_kpi.to_string())
    print(f"Simulazione test_9 completata. Dati salvati in: {excel_path}")

if __name__ == "__main__":
    test_9()
