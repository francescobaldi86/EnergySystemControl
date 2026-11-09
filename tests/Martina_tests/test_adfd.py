import energy_system_control as esc
from energy_system_control.components.explicit_components.producers import AnaerobicDigester, AnaerobicDigesterFromData
from energy_system_control.components.storage_units.thermal_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.components.base import ControlledComponent
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
import math, os, pytest
import pandas as pd
import os
from energy_system_control.constants.fluids import Methane, CarbonDioxide

DATA_PATH = "DATA"
FILENAME = "biogas_data_simulated_Lampedusa.csv"
BOILER_EFFICIENCY = 0.90
STORAGE_CAPACITY = 1.0
INITIAL_SOC = 0.50
SIMULATION_END_H = 168.0
TIME_STEP_H = 1 / 60
METHANE_COLUMN = "Contenuto metano in biogas prodotto [-]"
VOLUME_COLUMN = "Produzione biogas [l/h]"

class BoilerController(Controller):
    def _compute_action(self, state):
        return {k: 1.0 for k in self.controlled_component_names}

class FakeHeatSink(ControlledComponent):
    def __init__(self, name):
        self.heat_input_port_name = f"{name}_heat_input_port" 
        super().__init__(name, {self.heat_input_port_name: "heat"})
    def step(self, state, action):
        pass

def calculate_reference_power():
    df = pd.read_csv(os.path.join(DATA_PATH, FILENAME),sep=";",decimal=".",index_col=0)
    volume_flow = df[VOLUME_COLUMN].mean()
    methane_fraction = df[METHANE_COLUMN].mean()
    density = (methane_fraction* Methane.rho + (1 - methane_fraction) * CarbonDioxide.rho)
    methane_mass_fraction = (methane_fraction * Methane.rho / density)
    biogas_lhv = methane_mass_fraction * Methane.LHV
    mass_flow = volume_flow / 3_600_000 * density
    chemical_power = mass_flow * biogas_lhv
    return chemical_power * BOILER_EFFICIENCY

def run_biogas_test(boiler_power):
    
    components = [AnaerobicDigesterFromData(name="anaerobic_digester",data_path=DATA_PATH,filename=FILENAME,time_alignment="yearly",date_format="%Y-%m-%d %H:%M:%S",volume_flow_column_name=VOLUME_COLUMN,methane_fraction_column_name=METHANE_COLUMN,),
        BiogasStorage(name="biogas_storage",capacity=STORAGE_CAPACITY,initial_soc=INITIAL_SOC,initial_methane_fraction=0.57,),
        GasBoiler(name="gas_boiler",efficiency=BOILER_EFFICIENCY,max_power=boiler_power,source_type="biogas",),
        FakeHeatSink(name="heat_sink"),]
    controllers = [BoilerController("boiler_controller",["gas_boiler"],{})]
    sensors = [esc.SOCSensor("storage_SOC_sensor", "biogas_storage")]
    connections = [("anaerobic_digester_biogas_port", "biogas_storage_input"),("biogas_storage_output", "gas_boiler_biogas_input_port"),("gas_boiler_heat_output_port", "heat_sink_heat_input_port"),]
    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections,)
    sim_config = esc.SimulationConfig(simulation_end_h=SIMULATION_END_H,time_step_h=TIME_STEP_H,simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    return env, df_ports, df_sensors

def test_biogas_storage_slightly_increasing():

    reference_power = calculate_reference_power()
    boiler_power = reference_power * 0.98
    env, df_ports, df_sensors = run_biogas_test(boiler_power)
    storage = env.components["biogas_storage"]
    digester = env.components["anaerobic_digester"]
    boiler_mass_flow = df_ports["gas_boiler_biogas_input_port:mass"]

    print("\nTEST 1 - PRODUZIONE LEGGERMENTE SUPERIORE")
    print(f"Potenza di riferimento: {reference_power:.6f} kW")
    print(f"Potenza caldaia: {boiler_power:.6f} kW")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale: {storage.SOC:.4%}")

    assert boiler_mass_flow.abs().max() > 0.0
    assert storage.SOC > INITIAL_SOC
    assert storage.SOC < 1.0
test_biogas_storage_slightly_increasing()

def test_biogas_storage_filling_completely():
    reference_power = calculate_reference_power()
    boiler_power = reference_power * 0.10
    env, df_ports, df_sensors = run_biogas_test(boiler_power)
    storage = env.components["biogas_storage"]
    boiler_mass_flow = df_ports["gas_boiler_biogas_input_port:mass"]

    print("\nTEST 2 - RIEMPIMENTO COMPLETO")
    print(f"Potenza caldaia: {boiler_power:.6f} kW")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale: {storage.SOC:.4%}")

    assert boiler_mass_flow.abs().max() > 0.0
    assert math.isclose(storage.SOC, 1.0, abs_tol=1e-6)
test_biogas_storage_filling_completely()

def test_biogas_storage_empty_with_boiler():

    reference_power = calculate_reference_power()
    boiler_power = reference_power * 2.00
    env, df_ports, df_sensors = run_biogas_test(boiler_power)
    storage = env.components["biogas_storage"]
    boiler_mass_flow = df_ports["gas_boiler_biogas_input_port:mass"]
    boiler_heat_output = -df_ports["gas_boiler_heat_output_port:heat"]

    print("\nTEST 3 - SVUOTAMENTO STORAGE")
    print(f"Potenza caldaia: {boiler_power:.6f} kW")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale: {storage.SOC:.4%}")
    print(f"Massa finale: {storage.total_mass:.8f} kg")

    assert boiler_mass_flow.abs().max() > 0.0
    assert math.isclose(storage.total_mass,0.0,abs_tol=1e-6)
    assert math.isclose(storage.volume_m3,0.0,abs_tol=1e-6)
    assert math.isclose(boiler_heat_output.iloc[-1],0.0,abs_tol=1e-6)
test_biogas_storage_empty_with_boiler()
