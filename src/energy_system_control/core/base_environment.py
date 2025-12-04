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
            elif isinstance(component, Utility):
                self.components_classified['Utility'].append(component)
            elif isinstance(component, StorageUnit):
                self.components_classified['StorageUnit'].append(component)
            component.attach(get_environmental_data=self.get_environmental_data)
    
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
        # We also create a registry for each sensor
        for sensor_name in self.sensors.keys():
            self.signal_registry_sensors.register(sensor_name, "")
        
    def connect_ports(self):
        for connection in self.connections:
            self.ports[connection[0]].connect_port(connection[1])
            self.ports[connection[1]].connect_port(connection[0])

    def load_components_and_sensors_to_controllers(self):
        for name, controller in self.controllers.items():
            controller.load_controlled_components(self.components)
            controller.load_sensors(self.sensors)

    def read_timeseries_data(self):
        for _, component in self.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.time_step, self.time_end)

    def get_cumulated_electricity(self, port_name: str, unit: str = 'kWh', sign: str = 'net'):
        match unit:
            case 'kWh':
                scaling_factor = 1 / 3600
            case 'MWh':
                scaling_factor = 1 / 3_600_000
        match sign:
            case 'net':
                return self.get_cumulated_result(port_name, 'electricity', scaling_factor)
            case 'only positive' | 'only negative':
                return self.get_cumulated_result_with_sign(port_name, 'electricity', scaling_factor, sign)
    
    def get_DHW_temperature_comfort_index(self, port_name, boundary):
        # Measures the temperature-based comfort given a condition
        condition = abs(self.simulation_data.ports[port_name].flow['mass']) > 1e-6
        return sum(self.simulation_data.ports[port_name].T[condition] >= boundary) / len(self.simulation_data.ports[port_name].T[condition])

    def get_cumulated_result(self, port_name: str, layer_name: str, scaling_factor: float = 1):
        # Calculates the cumulated value of a given flow over the duration of the simulation
        return self.simulation_data.ports[:, self.signal_registry_ports.col_index(port_name, layer_name)].sum() * self.time_step * scaling_factor

    def get_cumulated_result_with_sign(self, port_name: str, layer_name: str, scaling_factor: float = 1, sign: str = 'only_positive'):
        # Calculates the cumulated value of a given flow over the duration of the simulation
        # Keeps only positive or negative values
        match sign:
            case "only positive":
                return self.simulation_data.ports[self.simulation_data.ports[:, self.signal_registry_ports.col_index(port_name, layer_name)] >=0, self.signal_registry_ports.col_index(port_name, layer_name)].sum() * self.time_step * scaling_factor
            case "only negative":
                return -self.simulation_data.ports[self.simulation_data.ports[:, self.signal_registry_ports.col_index(port_name, layer_name)] <=0, self.signal_registry_ports.col_index(port_name, layer_name)].sum() * self.time_step * scaling_factor
    
    def get_boundary_index(self, sensor_name: str, boundary: float, condition: str):
        # Calculates the fraction of time over the simulation a certain value was above or below a certain boundary
        match condition:
            case "gt" | ">" | ">=":
                return sum(self.simulation_data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")] >= boundary) / len(self.simulation_data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")])
            case "lt" | "<" | "<=":
                return sum(self.simulation_data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")] <= boundary) / len(self.simulation_data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")])