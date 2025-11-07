from typing import Dict, List, Callable, Any
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