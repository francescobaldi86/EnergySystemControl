import energy_system_control as esc
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
from energy_system_control.components.storage_units.thermal_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.controlled_components.base import HeatSource
import math


class FakeController(Controller):

    def get_action(self, state):
        return {k: 0.0 for k in self.controlled_component_names}

class FakeBoiler(HeatSource):

    def get_efficiency(self, state):
        return None
    def get_heat_output(self, state):
        return None
    def step(self, state, action):
        self.ports[self.power_input_port_name].flows['chemical_energy'] = 0.0


def test_biogas_production_and_storage():

    # COMPONENTS
    components = [
        AnaerobicDigester(name="anaerobic_digester",biogas_rate=10.0,methane_fraction=0.60),
        BiogasStorage(name="biogas_storage",capacity=500.0,initial_ch4_mass=50.0,initial_co2_mass=30.0),
        FakeBoiler(name='boiler', source_type='biogas')
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

    assert storage.ch4_mass > 50.0
    assert storage.co2_mass > 30.0
    assert storage.total_mass > 80.0

    assert storage.total_mass <= storage.capacity

test_biogas_production_and_storage()