import energy_system_control as esc
from energy_system_control.components.explicit_components.producers import AnaerobicDigester
from energy_system_control.components.storage_units.thermal_storage import BiogasStorage
from energy_system_control.components.controlled_components.other_heat_sources import GasBoiler
from energy_system_control.controllers.base import Controller
from energy_system_control.components.controlled_components.base import HeatSource
from energy_system_control.components.base import ControlledComponent
import math


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

test_biogas_production_and_storage()