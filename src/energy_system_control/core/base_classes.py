from typing import Dict, List, Callable, Any
from collections import defaultdict

class Node:
    name: str
    time: float
    time_id: int
    layers: List[str]
    flow: Dict[str, float]
    def __init__(self, name, layers):
        self.name = name
        self.time = 0.0
        self.time_id = 0
        self.layers = layers
        self.reset_flow_data()  # Sets each 

    def balance(self, flows):
        raise NotImplementedError
    
    def reset_flow_data(self):
        self.flow = {name: 0.0 for name in self.layers}


class Sensor():
    name: str
    current_measurement: float
    def __init__(self, name):
        self.name = name

    def get_measurement(self):
        return self.current_measurement
    
    def initialize(self):
        self.current_measurement = None

    def reset(self):
        self.current_measurement = None