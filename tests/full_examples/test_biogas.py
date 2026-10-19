import energy_system_control as esc
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
from energy_system_control.components.storage_units.thermal_storage import BiogasStorageFromData
from energy_system_control.components.storage_units.thermal_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.components.base import ControlledComponent
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
import math
import os 

class FakeController(Controller):

    def get_action(self, state):
        return {k: 0.0 for k in self.controlled_component_names}

class FakeBoiler(ControlledComponent):
    def __init__(self, name, consumption: float = 0.0):
        self.biogas_input_port_name = f'{name}_biogas_input_port'
        self.biogas_consumption = consumption / 3600 # Biogas consumption, in [m3/s]. The input value is in m3/h
        super().__init__(name, {self.biogas_input_port_name: 'biogas'})

    def step(self, state, action):
        self.ports[self.biogas_input_port_name].flows['volume'] = self.biogas_consumption * action
        self.ports[self.biogas_input_port_name].flows['mass'] = self.ports[self.biogas_input_port_name].flows['volume'] * self.ports[self.biogas_input_port_name].density  # m3/s * kg/m3 --> kg/s
        self.ports[self.biogas_input_port_name].flows['chemical_energy'] = self.ports[self.biogas_input_port_name].flows['mass'] * self.ports[self.biogas_input_port_name].LHV  # kg/s * kJ/kg --> kW
        


def test_biogas_production_and_storage():

    # COMPONENTS
    components = [
        AnaerobicDigester(name="anaerobic_digester",biogas_rate=10.0,methane_fraction=0.60),
        BiogasStorage(name="biogas_storage",capacity=500.0,initial_ch4_mass=50.0,initial_co2_mass=30.0),
        FakeBoiler(name='boiler')
        ]

    controllers = [
        FakeController('fake_controller', ['boiler'], {})
    ]
    sensors = []

    # CONNECTIONS
    connections = [("anaerobic_digester_biogas_port","biogas_storage_input"),
                   ("boiler_biogas_input_port","biogas_storage_output")]

    # ENVIRONMENT
    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections)

    sim_config = esc.SimulationConfig(time_start_h=0.0,time_end_h=1.0,time_step_h=1/60)

    sim = esc.Simulator(env, sim_config)

    results = sim.run()

    df_ports, df_controllers, df_sensors = results.to_dataframe()

    print("\n PORT RESULTS")
    print(df_ports)

    storage = env.components["biogas_storage"]

    print("\n FINAL STORAGE")
    print(f"CH4 mass  : {storage.ch4_mass:.4f} kg")
    print(f"CO2 mass  : {storage.co2_mass:.4f} kg")
    print(f"Total mass: {storage.total_mass:.4f} kg")
    print(f"Density       : {storage.biogas_density:.4f} kg/m3")
    print(f"Volume        : {storage.volume:.4f} m3")
    print(f"Capacity      : {storage.capacity:.4f} m3")
    print(f"Fill fraction : {storage.volume / storage.capacity:.4%}")

    assert storage.ch4_mass > 50.0
    assert storage.co2_mass > 30.0
    assert storage.total_mass > 80.0

    assert storage.volume <= storage.capacity

#test_biogas_production_and_storage()


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
    sensors = []
    
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
 
#test_biogas_storage_empty_with_boiler()   

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_uno():
    components = [
        # Domanda di acqua calda sanitaria
        esc.IEAHotWaterDemand(name="demand_DHW",reference_temperature=40,profile_name="M",),
        # Produzione di biogas
        AnaerobicDigester(name="anaerobic_digester",biogas_rate=1.5,methane_fraction=0.60,),
        # Accumulo di biogas con dati temporali
        BiogasStorageFromData(name="biogas_storage",capacity=100.0,data_path=__HERE__,filename="biogas_data.csv",initial_ch4_mass=0.0,initial_co2_mass=0.0,),
        # Caldaia alimentata a biogas
        GasBoiler(name="gas_boiler",efficiency=0.90,max_power=10.0,),
        # Accumulo di acqua calda
        esc.HotWaterStorage(name="hot_water_storage",max_temperature=80,tank_volume=200,T_0=45,convection_coefficient_losses=0.0,),
        # Rete di acqua fredda
        esc.ColdWaterGrid(name="water_grid",utility_type="fluid",),]

    controllers = [esc.HeaterControllerWithBandwidth("gas_boiler_controller","gas_boiler","storage_tank_temperature_sensor",40,10,)]

    sensors = [esc.TankTemperatureSensor("storage_tank_temperature_sensor","hot_water_storage",)]

    connections = [
        # Domanda e accumulo di acqua calda
        ("demand_DHW_fluid_port","hot_water_storage_hot_water_output_port",),
        # Caldaia e accumulo termico
        ("gas_boiler_heat_output_port","hot_water_storage_main_heat_input_port",),
        # Digestore e accumulo di biogas
        ("anaerobic_digester_biogas_port","biogas_storage_input",),
        # Accumulo di biogas e caldaia
        ("biogas_storage_output","gas_boiler_biogas_input_port",),
        # Ingresso dell'acqua fredda
        ("hot_water_storage_cold_water_input_port","water_grid_fluid_port",),]

    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections,)

    sim_config = esc.SimulationConfig(time_start_h=0.0,time_end_h=24.0 * 7,time_step_h=0.5,)

    sim = esc.Simulator(env, sim_config)

    results = sim.run()

    df_ports, df_controllers, df_sensors = results.to_dataframe()

    # Verifica della temperatura dell'accumulo
    assert math.isclose(df_sensors.loc[10.0, "storage_tank_temperature_sensor"],325,abs_tol=1,)

    df_ports.to_csv(os.path.join(__HERE__,"test_1_results_ports.csv",),sep=";",)
test_uno()
