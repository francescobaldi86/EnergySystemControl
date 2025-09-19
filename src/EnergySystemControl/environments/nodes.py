from dataclasses import dataclass
from EnergySystemControl.environments.base_classes import Node


class DynamicNode(Node):
    def __init__(self, name, inertia, starting_state: float = 0.0):
        super().__init__(name)
        self.state_variable = starting_state
        self.inertia = inertia

    def update(self, inflows, outflows, dt):
        net_flow = sum(inflows) - sum(outflows)
        self.state_variable += (net_flow / self.inertia) * dt


class BalanceNode(Node):
    def __init__(self, name):
        super().__init__(name)

    def check_balance(self, tol=1e-6):
        return abs(self.delta) < tol
    

class ThermalNode(DynamicNode):
    def __init__(self, name: str, inertia: float, T_0: float):
        super().__init__(name, inertia=inertia, starting_state=T_0)
    @property
    def T(self): 
        return self.state_variable
    @T.setter
    def T(self, value):
        self.state_variable = value


class ElectricalNode(BalanceNode):
    def __init__(self, name):
        super().__init__(name)

class ElectricalStorageNode(DynamicNode):
    max_capacity: float
    def __init__(self, name, max_capacity):
        super().__init__(name, inertia=1)
        self.max_capacity = max_capacity
    @property
    def SOC(self): 
        return self.state_variable / self.max_capacity
    @SOC.setter
    def SOC(self, value):
        self.state_variable = value * self.max_capacity


class MassNode(DynamicNode):
    max_capacity: float
    def __init__(self, name, max_capacity, m_0: float):
        super().__init__(name, inertia = 1)
        self.max_capacity = max_capacity
    @property
    def m(self): 
        return self.state_variable
    @m.setter
    def m(self, value):
        self.state_variable = value
