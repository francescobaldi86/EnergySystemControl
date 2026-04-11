from typing import Dict, List


class Node:
    name: str
    layers: List[str]
    flow: Dict[str, float]
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers
        self.reset_flow_data()  # Sets each 

    def balance(self, flows):
        raise NotImplementedError
    
    def reset_flow_data(self):
        self.flow = {name: 0.0 for name in self.layers}