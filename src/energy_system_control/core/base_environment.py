from typing import Dict, List
from collections import defaultdict
import pandas as pd
from energy_system_control.core.base_classes import *
from energy_system_control.core.port import *
from energy_system_control.core.node import *
from energy_system_control.sim.simulation_data import *
from energy_system_control.core.registry import *
from energy_system_control.components.base import *
from energy_system_control.components.explicit_components.demands import *
from energy_system_control.components.explicit_components.producers import *
from energy_system_control.components.storage_units.thermal_storage import *
from energy_system_control.components.storage_units.electric_storage import *
from energy_system_control.sensors.sensors import *
from energy_system_control.controllers.base import *
from energy_system_control.controllers.predictors import *
from energy_system_control.controllers.RL.RLcontrollers import *
from energy_system_control.helpers import *
from energy_system_control.io.data_provider import EnvironmentalDataProvider
from energy_system_control.sim.config import SimulationConfig




class Environment:
    def __init__(self, 
                 components: List[Component] = [], 
                 controllers: List[Controller] = [], 
                 sensors: List[Sensor] = [], 
                 connections = [], 
                 predictors: List[Predictor] = [],
                 environmental_data_provider: EnvironmentalDataProvider | None = None,
                 latitude: float | None = None,
                 longitude: float | None = None
                 ):
        self.nodes: Dict[str, Node] = {}
        self.ports: Dict[str, Port] = {}
        self.connections: List[tuple] = connections
        self.components: Dict[str, Component] = {component.name: component for component in components}
        self.components_classified = defaultdict(list)
        self.controllers: Dict[str, Controller] = {controller.name: controller for controller in controllers}
        self.ordered_controllers: List[str] = [controller.name for controller in controllers]
        self.sensors: Dict[str, Sensor] = {sensor.name: sensor for sensor in sensors}
        self.predictors: Dict[str, Predictor] = {predictor.name: predictor for predictor in predictors}
        self.environmental_data_provider = environmental_data_provider
        self.latitude = latitude
        self.longitude = longitude
        self.signal_registry_ports = SignalRegistry()
        self.signal_registry_controllers = SignalRegistry()
        self.signal_registry_sensors = SignalRegistry()
        # Ordering data
        self.load_internal_components()
        self.classify_components()
        self.create_ports()
        self.connect_ports()
        self.check_unconnected_ports()

    def load_internal_components(self):
        new_components = {}
        new_connections = []
        for component in self.components.values():
            if isinstance(component, CompositeComponent):
                new_components.update(component.get_internal_components())
                new_connections += component.get_internal_connections()
        self.components.update(new_components)
        self.connections += new_connections

    def initialize(self, state: SimulationState):
        if self.environmental_data_provider:
            if hasattr(self.environmental_data_provider, "initialize"):
                self.environmental_data_provider.initialize(state)
        self.create_data_registry()

    def add_component(self, component_name, component_type, **kwargs):
        if component_type not in Component.registry:
            raise ValueError(f"Unknown component type: {component_type}")
        cls = Component.registry[component_type]
        self.components[component_name] = cls(**kwargs)

    def classify_components(self, components: Dict[str, Component] | None = None):
        # Classify components based on their type
        dict_to_iterate = components if components else self.components
        for _, component in dict_to_iterate.items():
            if isinstance(component, ExplicitComponent):
                self.components_classified['ExplicitComponent'].append(component)
            elif isinstance(component, ControlledComponent):
                self.components_classified['ControlledComponent'].append(component)
            elif isinstance(component, ImplicitComponent):
                self.components_classified['ImplicitComponent'].append(component)
            elif isinstance(component, StorageUnit):
                self.components_classified['StorageUnit'].append(component)
            elif isinstance(component, Bus):
                self.components_classified['Bus'].append(component)
            elif isinstance(component, Grid):
                self.components_classified['Grid'].append(component)
            elif isinstance(component, CompositeComponent):
                self.components_classified['CompositeComponent'].append(component)
            else:
                raise ValueError(f'Component {component} is not classified')
    
    def create_ports(self, components: Dict[str, Component] | None = None):
        # Classify components based on their type
        dict_to_iterate = components if components else self.components
        for _, component in dict_to_iterate.items():
            self.ports.update(component.create_ports())

    def check_unconnected_ports(self):
        # This function checks whether there are ports that are not connected to any other port, and in case removes them
        for component in self.components.values():
            ports_to_remove = []
            for port_name, port in component.ports.items():
                if port.connected_port is None:
                    ports_to_remove.append(port_name)
            for port_to_remove in ports_to_remove:
                self.ports.pop(port_to_remove)
                component.ports.pop(port_to_remove)
                print(f'WARNING: Port {port_to_remove} was removed since it was not connected to any other port. The simulation will attempt to continue')

    
    def create_data_registry(self):
        # We create a registry for each pair port-layer
        for port_name, port in self.ports.items():
            for layer in port.layers:
                self.signal_registry_ports.register(port_name, layer)
                if isinstance(port, FluidPort):
                    self.signal_registry_ports.register(port_name, 'temperature')
        # We create a registry for each pair controller-component that will store the action
        for controller_name, controller in self.controllers.items():
            for component_name in controller.controlled_component_names:
                self.signal_registry_controllers.register(controller_name, component_name)
                if isinstance(controller, RLController):
                    self.signal_registry_controllers.register(controller_name, "reward")
                    self.signal_registry_controllers.register(controller_name, "td_error")
        # We also create a registry for each sensor
        for sensor_name in self.sensors.keys():
            self.signal_registry_sensors.register(sensor_name, "")

    def connect_ports(self):
        for connection in self.connections:
            self.ports[connection[0]].connect_port(self.ports[connection[1]])
            self.ports[connection[1]].connect_port(self.ports[connection[0]])

    def read_timeseries_data(self):
        for _, component in self.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.time_step, self.time_end)

