import energy_system_control as esc
from energy_system_control.components.explicit_components.producers import AnaerobicDigester, AnaerobicDigesterFromData
from energy_system_control.components.storage_units.thermal_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.components.base import ControlledComponent
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
import math, os, pytest

class BoilerController(Controller):

    def _compute_action(self, state):
        # Caldaia sempre accesa al 100%
        return {k: 1.0 for k in self.controlled_component_names}

class FakeHeatSink(ControlledComponent):

    def __init__(self, name):
        self.heat_input_port_name = f"{name}_heat_input_port"

        super().__init__(name,{self.heat_input_port_name: "heat"})

    def step(self, state, action):
        # Il calore viene semplicemente assorbito.Non ci interessa modellare il comportamento termico:serve solo a mantenere collegata l'uscita della caldaia.
        pass
        
def test_constant_biogas_production():

    components = [AnaerobicDigester(name="anaerobic_digester",biogas_rate=20.0, methane_fraction=0.57,),
        BiogasStorage(name="biogas_storage",capacity=100.0,initial_soc=0.40,initial_methane_fraction=0.57,),
        GasBoiler( name="gas_boiler",efficiency=0.90,max_power=20.0,source_type="biogas",),FakeHeatSink(name="heat_sink"),]

    controllers = [BoilerController("boiler_controller",["gas_boiler"],{},)]

    sensors = [esc.SOCSensor("biogas_storage_SOC_sensor","biogas_storage",)]

    connections = [("anaerobic_digester_biogas_port","biogas_storage_input",),
        ("biogas_storage_output","gas_boiler_biogas_input_port",),
        ("gas_boiler_heat_output_port","heat_sink_heat_input_port",),]

    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections,)

    sim_config = esc.SimulationConfig(simulation_end_h=168.0,time_step_h=1 / 60,)

    sim = esc.Simulator(env, sim_config)
    results = sim.run()

    df_ports, df_controllers, df_sensors = results.to_dataframe()

    storage = env.components["biogas_storage"]
    digester = env.components["anaerobic_digester"]

    print("\nRISULTATI FINALI")
    print(f"SOC iniziale: {storage.SOC_0:.4%}")
    print(f"SOC finale:   {storage.SOC:.4%}")
    print(f"Volume finale: {storage.volume:.4f} m3")
    print(f"Massa finale: {storage.total_mass:.4f} kg")

    # VERIFICHE
    assert math.isclose(storage.SOC_0,0.4,abs_tol=1e-9,)
    assert 0.0 <= storage.SOC <= 1.0
    assert storage.volume <= 1.0 + 1e-9
    assert digester.volume_flow_l_h == 20.0

    boiler_mass_flow = df_ports[ "gas_boiler_biogas_input_port:mass"]

    assert boiler_mass_flow.abs().max() > 0.0

    # Con produzione insufficiente e caldaia al 100%,lo storage non dovrebbe aumentare di SOC.
    assert storage.SOC <= 0.4