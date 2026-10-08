import energy_system_control as esc
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
from energy_system_control.components.storage_units.thermal_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.base import ControlledComponent

class BoilerController(Controller):

    def get_action(self, state):
        # Caldaia sempre accesa al 100%
        return {k: 1.0 for k in self.controlled_component_names}

class FakeHeatSink(ControlledComponent):

    def __init__(self, name):
        self.heat_input_port_name = f"{name}_heat_input_port"

        super().__init__(name,{self.heat_input_port_name: "heat"})

    def step(self, state, action):
        # Il calore viene semplicemente assorbito.
        # Non ci interessa modellare il comportamento termico:
        # serve solo a mantenere collegata l'uscita della caldaia.
        pass

def test_biogas_storage_empty_with_boiler():
    
    # COMPONENTS
    components = [AnaerobicDigester(name="anaerobic_digester",biogas_rate=0.0,methane_fraction=0.60),
                  BiogasStorage(name="biogas_storage",capacity=500.0,initial_ch4_mass=5.0,initial_co2_mass=3.0),
                  GasBoiler(name="boiler",efficiency=0.90,max_power=100.0,source_type="biogas"),
                  FakeHeatSink(name="heat_sink")]
                            
    # CONTROLLER
    controllers = [BoilerController("boiler_controller",["boiler"],{})]
    sensors = [esc.PowerSensor('boiler_heat_output_sensor', 'boiler_heat_output_port', 'heat')]
    
    # CONNECTIONS
    connections = [("anaerobic_digester_biogas_port","biogas_storage_input"),
                   ("boiler_biogas_input_port","biogas_storage_output"), 
                   ("boiler_heat_output_port","heat_sink_heat_input_port")]

    # ENVIRONMENT
    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections)

    # Simulazione di 1 ora
    sim_config = esc.SimulationConfig(
        time_start_h=0.0,
        time_end_h=1.0,
        time_step_h=1 / 60
    )

    sim = esc.Simulator(env, sim_config)
   
    results = sim.run()

    df_ports, df_controllers, df_sensors = results.to_dataframe()

    
    (-df_sensors['boiler_heat_output_sensor']).plot()

    print("\nPORT RESULTS")
    print(df_ports)

    storage = env.components["biogas_storage"]

    print("\nFINAL STORAGE")
    print(f"CH4 mass  : {storage.ch4_mass:.4f} kg")
    print(f"CO2 mass  : {storage.co2_mass:.4f} kg")
    print(f"Total mass: {storage.total_mass:.4f} kg")
    print(f"Density   : {storage.biogas_density:.4f} kg/m3")
    print(f"Volume    : {storage.volume:.4f} m3")
    print(f"Capacity  : {storage.capacity:.4f} m3")
    print(f"Fill      : {storage.volume / storage.capacity:.4%}")

    # 1
    # Lo storage deve essere completamente vuoto
    assert storage.total_mass == 0.0, (
        f"Storage non completamente vuoto: "
        f"{storage.total_mass:.4f} kg"
    )

    assert storage.volume == 0.0, (
        f"Storage non completamente vuoto: "
        f"{storage.volume:.4f} m3"
    )

    #2
    # La caldaia deve aver richiesto biogas
    boiler_mass_flow = df_ports[
        "boiler_biogas_input_port:mass"
    ]

    assert boiler_mass_flow.abs().max() > 0.0, (
        "La caldaia non ha richiesto biogas."
    )

    #3
    #La caldaia continua a richiedere biogas anche
    # quando lo storage è ormai vuoto.
    final_boiler_request = boiler_mass_flow.iloc[-1]

    assert final_boiler_request > 0.0, (
        "La caldaia non richiede più biogas."
    )

    # Questo assert deve FALLIRE.
    # Serve a evidenziare che la caldaia sta chiedendo
    # biogas mentre lo storage è vuoto.
    assert storage.total_mass > 0.0, (
        "ERRORE: la caldaia richiede biogas "
        "ma lo storage è vuoto."
    )
 
test_biogas_storage_empty_with_boiler()    
