from EnergySystemControl.environments.base_environment import Component
from typing import Dict, List


class Utility(Component):
    def __init__(self, name: str, nodes: List[str], project_path: str):
        super().__init__(self, name, project_path)


class HeatSource(Utility):
    def __init__(self, name: str, node: str, Q_W: float, project_path: str):
        super().__init__(self, name, project_path)
        self.node = node
        self.Q_W = Q_W

    def step(self, dt_s, nodes):
        Q_J = self.Q_W * dt_s
        return {self.node: Q_J}
    
class ElectricityGrid(Utility):
    def __init__(self, name: str, node: str, Wdot_max: float, project_path: str):
        super().__init__(self, name, project_path)
        self.node = node
        self.Wdot_max = Wdot_max

    def step(self, dt_s, nodes):
        Q_J = self.Q_W * dt_s
        return {self.node: Q_J}