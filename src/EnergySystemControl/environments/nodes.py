from dataclasses import dataclass

class Node:
    def __init__(self, name):
        self.name = name

    def balance(self, flows):
        raise NotImplementedError


class DynamicNode(Node):
    def __init__(self, name, state_variable, inertia):
        super().__init__(name)
        self.state_variable = state_variable
        self.inertia = inertia

    def update(self, inflows, outflows, dt):
        net_flow = sum(inflows) - sum(outflows)
        self.state_variable += (net_flow / self.inertia) * dt


class BalanceNode(Node):
    def __init__(self, name):
        super().__init__(name)

    def check_balance(self, inflows, outflows, tol=1e-6):
        net = sum(inflows) - sum(outflows)
        return abs(net) < tol
    

class ThermalNode(DynamicNode):
    @property
    def T(self): 
        return self.state_variable
    @T.setter
    def T(self, value):
        self.state_variable = value


class ElectricalNode(BalanceNode):
    def __init__(self, ):
        super().__init__()


class MassNode(DynamicNode):
    max_capacity: float
    def __init__(self, name, max_capacity):
        super().__init__(self, name)
        self.max_capacity = max_capacity
        self.inertia = 1
    @property
    def m(self): 
        return self.state_variable
    @m.setter
    def m(self, value):
        self.state_variable = value
