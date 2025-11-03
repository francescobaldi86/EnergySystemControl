from typing import Dict, List
from collections import defaultdict

class Node:
    def __init__(self, name):
        self.name = name
        self.time = 0.0
        self.time_id = 0
        self.flow = defaultdict(lambda: 0)

    def balance(self, flows):
        raise NotImplementedError
    
class Component:
    name: str
    time: float
    time_id: int
    registry = {}
    """Base class for components. Subclasses implement step(dt_s, nodes)."""
    def __init__(self, name: str, nodes: List[str]):
        self.name = name
        self.nodes = nodes
        self.time = 0.0
        self.time_id = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Component.registry[cls.__name__] = cls

    def step(self, time, time_step) -> Dict[str, float]:
        """
        Perform one time step.
        Returns a dict node_name -> heat_added_in_J (positive adds energy to node).
        """
        self.time = time
        self.time_step = time_step