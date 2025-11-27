from typing import Dict, List
from collections import defaultdict
import pandas as pd
from energy_system_control.core.base_classes import *
from energy_system_control.core.ports import *
from energy_system_control.core.registry import *
from energy_system_control.components.base import *
from energy_system_control.components.demands import *
from energy_system_control.components.producers import *
from energy_system_control.components.utilities import *
from energy_system_control.components.storage import *
from energy_system_control.sensors.sensors import *
from energy_system_control.controllers.base import *
from energy_system_control.helpers import *




class Environment:
    def __init__(self, components: List[Component] = [], controllers: List[Controller] = [], sensors: List[Sensor] = [], connections = []):
        self.nodes: Dict[str, Node] = {}
        self.ports: Dict[str, Port] = {}
        self.connections: List[tuple] = connections
        self.balance_nodes = {}
        self.dynamic_nodes = {}
        self.components: Dict[str, Component] = {component.name: component for component in components}
        self.components_classified = defaultdict(list)
        self.controllers: Dict[str, Controller] = {controller.name: controller for controller in controllers}
        self.ordered_controllers: List[str] = [controller.name for controller in controllers]
        self.sensors: Dict[str, Sensor] = {sensor.name: sensor for sensor in sensors}
        self.signal_registry_ports = SignalRegistry()
        self.signal_registry_controllers = SignalRegistry()
        self.signal_registry_sensors = SignalRegistry()
        self.simulation_data = SimulationData()
        # Ordering data
        self.classify_components()
        self.create_ports()
        self.connect_ports()
        self.load_components_and_sensors_to_controllers()
        self.create_data_registry()

    def add_component(self, component_name, component_type, **kwargs):
        if component_type not in Component.registry:
            raise ValueError(f"Unknown component type: {component_type}")
        cls = Component.registry[component_type]
        self.components[component_name] = cls(**kwargs)

    def classify_components(self):
        # Classify components based on their type
        for _, component in self.components.items():
            if isinstance(component, Demand):
                self.components_classified['Demand'].append(component)
            elif isinstance(component, Producer):
                self.components_classified['Producer'].append(component)
            elif isinstance(component, BalancingUtility):
                self.components_classified['BalancingUtility'].append(component)
            elif isinstance(component, Utility):
                self.components_classified['Utility'].append(component)
            elif isinstance(component, StorageUnit):
                self.components_classified['StorageUnit'].append(component)
            component.attach(get_environmental_data=self.get_environmental_data)
    
    def create_ports(self):
        for _, component in self.components.items():
            self.ports.update(component.create_ports())
    
    def create_data_registry(self):
        # We create a registry for each pair port-layer
        for port_name, port in self.ports.items():
            for layer in port.layers:
                self.signal_registry_ports.register(port_name, layer)
        # We create a registry for each pair controller-component that will store the action
        for controller_name, controller in self.controllers.items():
            for component_name in controller.controlled_component_names:
                self.signal_registry_controllers.register(controller_name, component_name)
        # We also create a registry for each sensor
        for sensor_name in self.sensors.keys():
            self.signal_registry_sensors.register(sensor_name, "")
        
    def connect_ports(self):
        for connection in self.connections:
            self.ports[connection[0]].connect_port(connection[1])
            self.ports[connection[1]].connect_port(connection[0])
    
    def get_environmental_data(self):
        return self.environmental_data

    def load_components_and_sensors_to_controllers(self):
        for name, controller in self.controllers.items():
            controller.load_controlled_components(self.components)
            controller.load_sensors(self.sensors)

    def read_timeseries_data(self):
        for _, component in self.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.time_step, self.time_end)

    def run(self, time_start: float = 0.0, time_end: float = 8760.0, time_step: float = 0.5):
        self.time_step = time_step * 3600  # Time step is stored in seconds
        # Read environmental data
        self.environmental_data = {}
        # Set the time step for all components once for all
        self.set_components_time_step()
        # Time
        self.time_start = time_start * 3600  # Time values are stored in seconds
        self.time_end = time_end * 3600  # Time values are stored in seconds
        self.time = time_start
        self.time_id = 0
        self.time_vector = np.arange(self.time_start, self.time_end, self.time_step)
        self.simulation_data.create_empty_datasets(self.time_vector, self.signal_registry_ports, self.signal_registry_controllers, self.signal_registry_sensors)
        self.read_timeseries_data()
        while self.time < self.time_end - 1e-9:
            self.step()
            self.time += self.time_step
            self.time_id += 1

    def step(self):
        """
        Taking a step involves the four main actions, executed in sequence:
        1. Simulate "demand" type components (they do not depend on other components' behaviour)
        2. Simulate "production" type components (they do not depend on other components' behaviour)
        3. Activate controllers
        4. Simulate "utility" type components (they require the respective controllers to execute)
        5. Update node states
        """
        # Put to 0 the delta of the balance at each node
        for _, port in self.ports.items():
            port.reset_flow_data()
        # Initialize control actions
        self.control_actions = defaultdict(lambda: None)
        self.components_to_simulate = [x for x in self.components.keys()]
        # Simulate components
        self.update_environmental_data()
        # Assign fluid port values
        self.propagate_fluid_port_values()
        self.simulate_components_of_type('Demand')
        self.simulate_components_of_type('Producer')
        self.get_controller_actions()
        self.simulate_components_of_type('Utility')
        self.simulate_components_of_type('StorageUnit')
        self.simulate_components_of_type('BalancingUtility')
        if self.components_to_simulate != []:
            raise(BaseException, f'The step was concluded but components {self.components_to_simulate} where not simulated at time {self.time}, time ID {self.time_id}. Check what happened!')
        self.save_simulation_data()

    def update_environmental_data(self):
        self.environmental_data = {
            'Temperature cold water': C2K(15),
            'Temperature ambient': C2K(20)
        }
    
    def propagate_fluid_port_values(self):
        for _, component in self.components.items():
            port_name, T = component.set_inherited_fluid_port_values()  # Sets the value for each port that can do so
            if port_name:  # Propagates the value to the connected port
                self.ports[self.ports[port_name].connected_port].T = T

    def simulate_components_of_type(self, type: str):
        components = self.components_classified[type]
        for component in components:
            if component.name in self.components_to_simulate:
                component.time = self.time
                component.time_id = self.time_id
                self.take_component_step(component, None)           
                self.components_to_simulate.remove(component.name)

    def get_controller_actions(self):
        for controller_name in self.ordered_controllers:
            self.controllers[controller_name].time = self.time
            self.controllers[controller_name].time_id = self.time_id
            self.controllers[controller_name].get_obs(self)
            actions = self.controllers[controller_name].get_action()
            # After a controller has been simulated, its controlled component is immediately simulted as well
            for component_name, action in actions.items():
                if component_name in self.components_to_simulate:
                    self.take_component_step(self.components[component_name], action)
                    self.components_to_simulate.remove(component_name)
                else:
                    raise(KeyError, f'Component {component_name} has been simulated before its related control action was calculated at time {self.time}, time ID {self.time_id}. Check what happens!')
            self.control_actions.update(actions)

    def take_component_step(self, component, action):
        component.step(action)
        # Update values of connected ports
        for _, port in component.ports.items():
            for layer, value in port.flow.items():
                if port.connected_port:
                    self.ports[port.connected_port].flow[layer] = -value
                    if isinstance(self.ports[port.connected_port], FluidPort):
                        self.ports[port.connected_port].T = self.ports[port.name].T

    def save_simulation_data(self):
        # Ports
        for port_name, port in self.ports.items():
            for layer, flow in port.flow.items():
                col = self.signal_registry_ports.col_index(port_name, layer)
                self.simulation_data.ports[self.time_id, col] = flow
        # Controllers
        for controller_name, controller in self.controllers.items():
            for component_name in controller.controlled_component_names:
                col = self.signal_registry_controllers.col_index(controller_name, component_name)
                self.simulation_data.controllers[self.time_id, col] = controller.previous_action[component_name]
        # We also create a registry for each sensor
        for sensor_name, sensor in self.sensors.items():
            col = self.signal_registry_sensors.col_index(sensor_name, "")
            self.simulation_data.sensors[self.time_id, col] = sensor.current_measurement

    def to_dataframe(self):
        return self.simulation_data.to_dataframe(self.time_vector, self.signal_registry_ports, self.signal_registry_controllers, self.signal_registry_sensors)
    
    def set_components_time_step(self):
        for _, component in self.components.items():
            component.time_step = self.time_step
        for _, controller in self.controllers.items():
            controller.time_step = self.time_step
        for _, sensor in self.sensors.items():
            sensor.time_step = self.time_step