from typing import Dict, List, Callable, Any
from collections import defaultdict

class Node:
    name: str
    time: float
    time_id: int
    flow: dict
    delta: float
    def __init__(self, name):
        self.name = name
        self.time = 0.0
        self.time_id = 0
        self.delta = 0.0
        self.flow = defaultdict(lambda: 0)

    def balance(self, flows):
        raise NotImplementedError
    
    def reset_flow_data(self):
        self.flow = defaultdict(lambda: 0)


class Component:
    name: str
    time: float
    time_id: int
    time_step: float
    # environment: esc.Environment
    registry = {}
    node_names: List[str]
    nodes: Dict[str, Node]
    """Base class for components. Subclasses implement step(dt_s, nodes)."""
    def __init__(self, name: str, nodes: List[str]):
        self.name = name
        self.node_names = nodes
        self.time = 0.0
        self.time_id = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Component.registry[cls.__name__] = cls

    def attach(self, *, get_environmental_data: Callable[[str, Any], None]):
        self._environmental_data = get_environmental_data


class Sensor():
    def __init__(self, name):
        self.name = name

    def get_measurement(self, environment):
        raise(NotImplementedError)
    

class Controller():
    name: str
    time: float
    time_id: int
    time_step: float
    controlled_component_names: List[str]
    sensor_names: Dict[str, str]
    controlled_components: Dict[str, Component]
    sensors: Dict[str, Sensor]
    obs: dict
    def __init__(self, name, controlled_components: List[str], sensors: Dict[str, str]):
        """
        Class for a generic controller

        Parameters
        ----------
        name : str
            Name of the component
        controlled_components : list
            A list of the names of the controlled components
        sensors: dict
            A dictionary where each item corresponds to a sensor, and the respective key corresponds to the name of the variable read by the sensor 
        """
        self.name = name
        self.controlled_component_names = controlled_components
        self.sensor_names = sensors

    def get_obs(self, environment):
        self.obs = {var: sensor.get_measurement(environment) for var, sensor in self.sensors.items()}

    def load_controlled_components(self, components):
        self.controlled_components = {name: components[name] for name in self.controlled_component_names}
    
    def load_sensors(self, sensors):
        self.sensors = {var: sensors[sensor_name] for var, sensor_name in self.sensor_names.items()}

    def get_action(self):
        return None