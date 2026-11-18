from typing import Dict, List
from collections import defaultdict
from dataclasses import fields
import warnings
import numpy as np
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
from energy_system_control.sim.step_plan import StepPlan




class Environment:
    def __init__(self,
                 components: List[Component] = [],
                 controllers: List[Controller] = [],
                 sensors: List[Sensor] = [],
                 connections = [],
                 predictors: List[Predictor] = [],
                 environmental_data_provider: EnvironmentalDataProvider | None = None,
                 latitude: float | None = None,
                 longitude: float | None = None,
                 strict_unconnected_ports: bool = False
                 ):
        self.nodes: Dict[str, Node] = {}
        self.ports: Dict[str, Port] = {}
        self.connections: List[tuple] = list(connections)
        self.strict_unconnected_ports = strict_unconnected_ports
        self.components: Dict[str, Component] = self._dedup_named(components, 'component')
        self.components_classified = defaultdict(list)
        self.controllers: Dict[str, Controller] = self._dedup_named(controllers, 'controller')
        self.ordered_controllers: List[str] = [name_of(controller) for controller in controllers]
        self.sensors: Dict[str, Sensor] = self._dedup_named(sensors, 'sensor')
        self.predictors: Dict[str, Predictor] = self._dedup_named(predictors, 'predictor')
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

    @staticmethod
    def _dedup_named(objects, kind: str) -> Dict[str, Any]:
        """Build a name -> object dict, failing fast on duplicate names."""
        result = {}
        for obj in objects:
            name = name_of(obj)
            if name in result:
                raise ValueError(f"Duplicate {kind} name '{name}'. Names must be unique.")
            result[name] = obj
        return result

    def resolve_component(self, spec) -> Component:
        """Resolve a component given as a name (str) or a Component object."""
        if isinstance(spec, Component):
            if spec.name in self.components:
                return self.components[spec.name]
            raise KeyError(f"Component '{spec.name}' is not part of this environment. "
                           f"Available components: {sorted(self.components)}")
        name = str(spec)
        if name not in self.components:
            suggestions = close_matches(name, self.components.keys())
            msg = f"Unknown component '{name}'. Available components: {sorted(self.components)}"
            if suggestions:
                msg += f". Did you mean: {suggestions}?"
            raise KeyError(msg)
        return self.components[name]

    def resolve_sensor(self, spec) -> Sensor:
        """Resolve a sensor given as a name (str) or a Sensor object."""
        if isinstance(spec, Sensor):
            if spec.name in self.sensors:
                return self.sensors[spec.name]
            raise KeyError(f"Sensor '{spec.name}' is not part of this environment. "
                           f"Available sensors: {sorted(self.sensors)}")
        name = str(spec)
        if name not in self.sensors:
            suggestions = close_matches(name, self.sensors.keys())
            msg = f"Unknown sensor '{name}'. Available sensors: {sorted(self.sensors)}"
            if suggestions:
                msg += f". Did you mean: {suggestions}?"
            raise KeyError(msg)
        return self.sensors[name]

    def resolve_port(self, spec) -> Port:
        """Resolve a port given as a Port object, a full port-name string, a
        (component, role) tuple, or a single-port Component.

        Examples: ``env.resolve_port('heat_pump_heat_output_port')``,
        ``env.resolve_port((heat_pump, 'heat_output_port'))``,
        ``env.resolve_port(heat_pump.port('heat_output_port'))``.
        """
        if isinstance(spec, Port):
            if spec.name in self.ports and self.ports[spec.name] is spec:
                return spec
            raise KeyError(f"Port '{spec.name}' is not part of this environment. "
                           f"Available ports: {sorted(self.ports)}")
        if isinstance(spec, Component):
            if len(spec.ports) == 1:
                return next(iter(spec.ports.values()))
            raise ValueError(f"Component '{spec.name}' has {len(spec.ports)} ports; specify one as "
                             f"('{spec.name}', role). Its ports: {list(spec.ports.keys())}")
        if isinstance(spec, tuple) and len(spec) == 2:
            component = self.resolve_component(spec[0])
            return component.port(spec[1])
        name = str(spec)
        if name not in self.ports:
            suggestions = close_matches(name, self.ports.keys())
            msg = f"Unknown port '{name}'. Available ports: {sorted(self.ports)}"
            if suggestions:
                msg += f". Did you mean: {suggestions}?"
            raise KeyError(msg)
        return self.ports[name]

    def _normalize_connection(self, entry) -> tuple:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(
                f"Each connection must be a pair of two ports, got: {entry!r}. Accepted formats: "
                f"('port_a_name', 'port_b_name'), (port_a, port_b), or "
                f"(('component_a', 'role_a'), ('component_b', 'role_b')).")
        port_a = self.resolve_port(entry[0])
        port_b = self.resolve_port(entry[1])
        if port_a is port_b:
            raise ValueError(f"Cannot connect port '{port_a.name}' to itself.")
        return port_a, port_b

    def connect(self, port_a, port_b):
        """Connect two ports of this environment.

        Accepts Port objects, full port-name strings, (component, role) tuples,
        or single-port components, e.g.::

            env.connect(('heat_pump', 'heat_output_port'), ('hot_water_storage', 'main_heat_input_port'))
            env.connect(heat_pump.port('heat_output_port'), tank.port('main_heat_input_port'))
        """
        a, b = self._normalize_connection((port_a, port_b))
        pair = (a.name, b.name)
        if pair in self.connections or (pair[1], pair[0]) in self.connections:
            raise ValueError(f"Duplicate connection between '{a.name}' and '{b.name}'.")
        a.connect_port(b)
        b.connect_port(a)
        self.connections.append(pair)

    def load_internal_components(self):
        new_components = {}
        new_connections = []
        for component in self.components.values():
            if isinstance(component, CompositeComponent):
                new_components.update(component.get_internal_components())
                new_connections += component.get_internal_connections()
        self.components.update(new_components)
        self.connections += new_connections

    def initialize(self, state: SimulationState, cfg: SimulationConfig):
        if self.environmental_data_provider:
            if hasattr(self.environmental_data_provider, "initialize"):
                self.environmental_data_provider.initialize(state, cfg)
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
        # Checks whether there are ports that are not connected to any other port.
        # They are kept in the environment (so they stay visible in the results
        # and can still be connected later with env.connect(...)), but all of them
        # are reported together in a single loud warning — or an error when
        # strict_unconnected_ports=True.
        unconnected = []
        for component in self.components.values():
            for port_name, port in component.ports.items():
                if port.connected_port is None:
                    unconnected.append(port_name)
        if not unconnected:
            return
        message = (f"{len(unconnected)} port(s) are not connected to any other port: {unconnected}. "
                   f"Connect them via Environment(connections=[...]) or env.connect(port_a, port_b).")
        if self.strict_unconnected_ports:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    
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

    def build_step_plan(self, state: SimulationState, cfg: SimulationConfig) -> StepPlan:
        """Compile everything the per-timestep loop needs into plain lists.

        Must be called after ``initialize`` (the signal registries are created
        there). Registry column ``i`` always corresponds to the i-th entry of
        ``_col_to_key``, so the save plans below are aligned with the columns
        of the SimulationData arrays by construction.
        """
        reg_ports = self.signal_registry_ports
        reg_controllers = self.signal_registry_controllers
        reg_sensors = self.signal_registry_sensors

        # (port, layer) for every recorded port signal; for FluidPort the extra
        # 'temperature' registration reads port.T instead of port.flows.
        port_save = [(self.ports[key.main_key], key.secondary_key)
                     for key in reg_ports._col_to_key]

        # Sensors in column order.
        sensor_save = [self.sensors[key.main_key]
                       for key in reg_sensors._col_to_key]

        # Per controller: {controlled component name: column}, plus the RL
        # reward/td_error columns when present.
        controller_save = []
        for controller_name, controller in self.controllers.items():
            col_map: Dict[str, int] = {}
            reward_col = None
            td_error_col = None
            for col, key in enumerate(reg_controllers._col_to_key):
                if key.main_key != controller_name:
                    continue
                if key.secondary_key == "reward":
                    reward_col = col
                elif key.secondary_key == "td_error":
                    td_error_col = col
                else:
                    col_map[key.secondary_key] = col
            controller_save.append((controller, col_map, reward_col, td_error_col))

        # One (port_a, port_b, layer) triple per connection x layer of port_a.
        balance_checks = []
        for port_name_a, port_name_b in self.connections:
            port_a = self.ports[port_name_a]
            port_b = self.ports[port_name_b]
            for layer in port_a.layers:
                balance_checks.append((port_a, port_b, layer))

        # Per component: connected (port, connected_port, layers, copy_T) tuples
        # for _take_component_step. copy_T mirrors the old behavior of copying
        # the temperature onto FluidPort/HeatPort connections.
        step_propagation: Dict[str, list] = {}
        for component in self.components.values():
            entries = []
            for port in component.ports.values():
                connected = port.connected_port
                if connected is not None:
                    copy_T = isinstance(connected, (FluidPort, HeatPort))
                    entries.append((port, connected, list(port.layers), copy_T))
            step_propagation[component.name] = entries

        # One datetime per step (for the environmental data provider).
        datetimes = state.simulation_start_datetime + pd.to_timedelta(state.time_vector, unit="s")

        # Solar angles for the whole horizon, computed once (pvlib accepts a
        # DatetimeIndex). Only when the location is set.
        solar_zenith = None
        solar_azimuth = None
        if self.latitude and self.longitude:
            zenith, azimuth = calculate_solar_angles(self.latitude, self.longitude, datetimes)
            solar_zenith = np.asarray(zenith, dtype=float)
            solar_azimuth = np.asarray(azimuth, dtype=float)

        return StepPlan(
            ports=list(self.ports.values()),
            unconnected_ports=[port for port in self.ports.values() if port.connected_port is None],
            sensors=list(self.sensors.values()),
            controllers=[self.controllers[name] for name in self.ordered_controllers],
            inherit=list(self.components.values()),
            explicit=list(self.components_classified["ExplicitComponent"]),
            controlled=list(self.components_classified["ControlledComponent"]),
            algebraic=list(self.components_classified["Bus"]) + list(self.components_classified["ImplicitComponent"]),
            storage=list(self.components_classified["StorageUnit"]),
            grid=list(self.components_classified["Grid"]),
            step_propagation=step_propagation,
            port_save=port_save,
            controller_save=controller_save,
            sensor_save=sensor_save,
            balance_checks=balance_checks,
            datetimes=datetimes,
            env_field_names=[f.name for f in fields(EnvironmentalData)],
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
        )

    def connect_ports(self):
        # Normalize every connection entry (strings, Port objects, (component, role)
        # tuples) to a pair of port names, validating as we go, then connect.
        normalized = []
        seen = set()
        for entry in self.connections:
            port_a, port_b = self._normalize_connection(entry)
            pair = (port_a.name, port_b.name)
            key = frozenset(pair)
            if key in seen:
                raise ValueError(f"Duplicate connection between '{port_a.name}' and '{port_b.name}'.")
            seen.add(key)
            port_a.connect_port(port_b)   # raises ValueError on type mismatch or conflicting connection
            port_b.connect_port(port_a)
            normalized.append(pair)
        self.connections = normalized

    def read_timeseries_data(self):
        for _, component in self.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.time_step, self.time_end)

