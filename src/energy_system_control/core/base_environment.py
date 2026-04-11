from typing import Dict, List
from collections import defaultdict
import pandas as pd
from energy_system_control.core.base_classes import *
from energy_system_control.core.ports import *
from energy_system_control.sim.simulation_data import *
from energy_system_control.core.registry import *
from energy_system_control.components.base import *
from energy_system_control.components.demands import *
from energy_system_control.components.producers import *
from energy_system_control.components.utilities import *
from energy_system_control.components.storage import *
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
        self.classify_components()
        self.create_ports()
        self.connect_ports()
        

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
            if isinstance(component, Demand):
                self.components_classified['Demand'].append(component)
            elif isinstance(component, Producer):
                self.components_classified['Producer'].append(component)
            elif isinstance(component, BalancingUtility):
                self.components_classified['BalancingUtility'].append(component)
            elif isinstance(component, StorageUnit):
                self.components_classified['StorageUnit'].append(component)
            elif isinstance(component, HeatSource):
                self.components_classified['HeatSource'].append(component)
            else:
                self.components_classified['Other'].append(component)
    
    def create_ports(self, components: Dict[str, Component] | None = None):
        # Classify components based on their type
        dict_to_iterate = components if components else self.components
        for _, component in dict_to_iterate.items():
            self.ports.update(component.create_ports())
    
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
            self.ports[connection[0]].connect_port(connection[1])
            self.ports[connection[1]].connect_port(connection[0])

    def read_timeseries_data(self):
        for _, component in self.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.time_step, self.time_end)

