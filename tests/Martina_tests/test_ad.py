import energy_system_control as esc
import math
import pytest
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
from energy_system_control.components.storage_units.biogas_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.base import ControlledComponent

BOILER_POWER = 20.0
BOILER_EFFICIENCY = 0.90
METHANE_FRACTION = 0.57
METHANE_LHV = 9.97  # kWh/Nm3

def calculate_biogas_rate(multiplier=1.0):
    required_flow = (BOILER_POWER /(BOILER_EFFICIENCY * METHANE_FRACTION * METHANE_LHV))
    return required_flow * 1000 * multiplier

class BoilerController(Controller):
    def _compute_action(self, state):
        # Caldaia sempre accesa al 100%
        if self.obs['fuel_storage_SOC'] >= 0.1:
            return {k: 1.0 for k in self.controlled_component_names}
        else:
            return {k: 0.0 for k in self.controlled_component_names}
    
class FakeHeatSink(ControlledComponent):
    def __init__(self, name):
        self.heat_input_port_name = f"{name}_heat_input_port"
        super().__init__(name, {self.heat_input_port_name: "heat"})
    def step(self, state, action):
        pass

def test_biogas_storage_slightly_increasing():

    biogas_rate = calculate_biogas_rate(1.02)

    components = [AnaerobicDigester(name="anaerobic_digester",biogas_rate=biogas_rate,methane_fraction=METHANE_FRACTION),
        BiogasStorage(name="biogas_storage",capacity=1.0,initial_soc=0.50,initial_methane_fraction=METHANE_FRACTION),
        GasBoiler(name="gas_boiler",efficiency=BOILER_EFFICIENCY,max_power=BOILER_POWER,source_type="biogas"),
        FakeHeatSink(name="heat_sink")]
    
    controllers = [BoilerController("boiler_controller", ["gas_boiler"], {'fuel_storage_SOC': 'storage_SOC_sensor'})]
    sensors = [esc.SOCSensor("storage_SOC_sensor", "biogas_storage")]
    connections = [("anaerobic_digester_biogas_port", "biogas_storage_input"),
        ("biogas_storage_output", "gas_boiler_biogas_input_port"),
        ("gas_boiler_heat_output_port", "heat_sink_heat_input_port")]
    
    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections)
    sim_config = esc.SimulationConfig(simulation_end_h=2,time_step_h=1/60)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    storage = env.components["biogas_storage"]
    digester = env.components["anaerobic_digester"]
    print("\n--- VERIFICA BILANCIO ---")
    print("Portate digestore:")
    print(df_ports["anaerobic_digester_biogas_port:mass"])
    print("\nPortate caldaia:")
    print(df_ports["gas_boiler_biogas_input_port:mass"])
    print("\nFlussi energetici caldaia:")
    print(df_ports["gas_boiler_biogas_input_port:chemical_energy"])
    print("\nSOC registrato durante la simulazione:")
    print(df_sensors)
    simulation_time_h = 3 / 60
    produced_volume = biogas_rate * simulation_time_h
    expected_boiler_volume = (calculate_biogas_rate(1.0) * simulation_time_h)
    expected_net_volume = produced_volume - expected_boiler_volume
    expected_soc = (storage.SOC_0 + expected_net_volume / (storage.capacity * 1000))
    print(f"Volume prodotto: {produced_volume:.4f} L")
    print(f"Volume richiesto caldaia: {expected_boiler_volume:.4f} L")
    print(f"Volume netto atteso: {expected_net_volume:.4f} L")
    print(f"SOC atteso: {expected_soc:.6%}")
    print(f"SOC simulato: {storage.SOC:.6%}")
    print("\nTEST 1 - PRODUZIONE LEGGERMENTE SUPERIORE")
    print(f"Produzione digestore: {biogas_rate:.2f} L/h")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale: {storage.SOC:.4%}")
    print(f"Massa finale: {storage.total_mass:.4f} kg")

    boiler_mass_flow = df_ports["gas_boiler_biogas_input_port:mass"]
    assert math.isclose(digester.volume_flow_l_h,biogas_rate,abs_tol=1e-6)

    # La caldaia deve consumare biogas
    assert boiler_mass_flow.abs().max() > 0.0
    # Lo storage deve aumentare leggermente ma non raggiunge la capacità massima
    assert 1.0>storage.SOC > 0.50   



def test_biogas_storage_filling_completely():

    biogas_rate = calculate_biogas_rate(1.50)

    components = [AnaerobicDigester(name="anaerobic_digester",biogas_rate=biogas_rate,methane_fraction=METHANE_FRACTION),
        BiogasStorage(name="biogas_storage",capacity=1.000,initial_soc=0.50,initial_methane_fraction=METHANE_FRACTION),
        GasBoiler(name="gas_boiler",efficiency=BOILER_EFFICIENCY,max_power=BOILER_POWER,source_type="biogas"),
        FakeHeatSink(name="heat_sink")]
    controllers = [BoilerController("boiler_controller", ["gas_boiler"], {'fuel_storage_SOC': 'storage_SOC_sensor'})]
    sensors = [esc.SOCSensor("storage_SOC_sensor", "biogas_storage")]
    connections = [("anaerobic_digester_biogas_port", "biogas_storage_input"),
        ("biogas_storage_output", "gas_boiler_biogas_input_port"),
        ("gas_boiler_heat_output_port", "heat_sink_heat_input_port")]

    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections)
    sim_config = esc.SimulationConfig(simulation_end_h=168.0,time_step_h=1 / 60)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    storage = env.components["biogas_storage"]
    digester = env.components["anaerobic_digester"]
    boiler_mass_flow = df_ports["gas_boiler_biogas_input_port:mass"]

    print("\nTEST 2 - PRODUZIONE ECCESSIVA")
    print(f"Produzione digestore: {biogas_rate:.2f} L/h")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale: {storage.SOC:.4%}")
    print(f"Massa finale: {storage.total_mass:.4f} kg")

    assert math.isclose(digester.volume_flow_l_h,biogas_rate,abs_tol=1e-6)

    # La caldaia continua a consumare biogas
    assert boiler_mass_flow.abs().max() > 0.0

    # Lo storage deve raggiungere la capacità massima
    assert math.isclose(storage.SOC,1.0,abs_tol=1e-6)    


def test_biogas_storage_empty_with_boiler():

    biogas_rate = calculate_biogas_rate(0.0)
    storage = BiogasStorage(name="biogas_storage",capacity=300.0,initial_soc=0.05,initial_methane_fraction=0.60)
    components = [AnaerobicDigester(name="anaerobic_digester",biogas_rate=biogas_rate,methane_fraction=0.60),storage,GasBoiler(name="boiler",efficiency=0.90,max_power=100.0,source_type="biogas"),FakeHeatSink(name="heat_sink")]
    controllers = [BoilerController("boiler_controller", ["boiler"], {'fuel_storage_SOC': 'storage_SOC_sensor'}, storage)]
    sensors = [esc.SOCSensor("storage_SOC_sensor", "biogas_storage")]
    connections = [("anaerobic_digester_biogas_port", "biogas_storage_input"),
        ("boiler_biogas_input_port", "biogas_storage_output"),
        ("boiler_heat_output_port", "heat_sink_heat_input_port")]

    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections)
    sim_config = esc.SimulationConfig(simulation_end_h=168.0 ,time_step_h=1/60)
    sim = esc.Simulator(env, sim_config)
    results = sim.run()
    df_ports, df_controllers, df_sensors = results.to_dataframe()
    storage = env.components["biogas_storage"]
    digester = env.components["anaerobic_digester"]

    storage_port = storage.ports[storage.output_port_name]

    boiler_mass_flow = df_ports["boiler_biogas_input_port:mass"]
    boiler_heat_output = -df_ports["boiler_heat_output_port:heat"]

    print("\nTEST 3 - SVUOTAMENTO STORAGE")
    print(f"Produzione digestore: {biogas_rate:.2f} L/h")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale: {storage.SOC:.4%}")
    print(f"Massa finale: {storage.total_mass:.6f} kg")
    print(f"Volume finale: {storage.volume_m3:.6f} m3")
   
    assert math.isclose(digester.volume_flow_l_h,biogas_rate,abs_tol=1e-6)

    # La caldaia deve aver consumato biogas inizialmente
    assert boiler_mass_flow.abs().max() > 0.0

    # Lo storage deve essere completamente vuoto
    assert math.isclose(storage.total_mass,0.0,abs_tol=1e-6)

    assert math.isclose(storage.volume_m3,0.0,abs_tol=1e-6)

    # La caldaia deve essersi spenta
    assert math.isclose(boiler_heat_output.iloc[-1],0.0,abs_tol=1e-6)    


if __name__ == "__main__":
    test_biogas_storage_empty_with_boiler()
    test_biogas_storage_filling_completely()
    test_biogas_storage_slightly_increasing()