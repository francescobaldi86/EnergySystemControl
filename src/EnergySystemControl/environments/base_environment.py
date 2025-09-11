from typing import Dict, List
import pandas as pd
from EnergySystemControl.environments.demands import *
from EnergySystemControl.environments.producers import *
from EnergySystemControl.environments.utilities import *
from EnergySystemControl.environments.storage_units import *
from EnergySystemControl.environments.nodes import *
from EnergySystemControl.helpers import *

    
class Component:
    name: str
    project_path: str
    time: float
    """Base class for components. Subclasses implement step(dt_s, nodes)."""
    def __init__(self, name: str, nodes: List[Node], project_path: str):
        self.name = name
        self.project_path = project_path
        self.nodes = nodes
        self.time = 0

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

class Environment:
    def __init__(self, nodes: Dict[str, Node], components: Dict[str, Component], controllers, dt_s: float):
        self.nodes = nodes
        self.balance_nodes = {}
        self.dynamic_modes = {}
        self.components = components
        self.demand_type_components = {}
        self.producer_type_components = {}
        self.utility_type_components = {}
        self.storage_type_components = {}
        self.controllers = controllers
        self.dt_s = dt_s
        # Data collection
        self.time = []
        self.history = {n: [] for n in nodes.keys()}
        self.comp_history = {c.name: [] for c in components}
        self.classify_nodes()

    def add_component(self, component_name, component_type, **kwargs):
        if component_type not in Component.registry:
            raise ValueError(f"Unknown component type: {component_type}")
        cls = Component.registry[component_type]
        self.components[component_type] = cls(**kwargs)
        

    def classify_nodes(self):
        # Classify nodes based on whether they are balance or dynamic nodes
        for name, node in self.nodes:
            if isinstance(node, BalanceNode):
                self.balance_nodes[name] = node
            elif isinstance(node, DynamicNode):
                self.dynamic_modes[name] = node

    def classify_components(self):
        # Classify components based on their type
        for name, component in self.components:
            if isinstance(component, Demand):
                self.demand_type_components[name] = component
            elif isinstance(component, Producer):
                self.producer_type_components[name] = component
            elif isinstance(component, Utility):
                self.utility_type_components[name] = component
            elif isinstance(component, StorageUnit):
                self.storage_type_components[name] = component

    def step(self, t_s: float):
        """
        Taking a step involves the four main actions, executed in sequence:
        1. Simulate "demand" type components (they do not depend on other components' behaviour)
        2. Simulate "production" type components (they do not depend on other components' behaviour)
        3. Activate controllers
        4. Simulate "utility" type components (they require the respective controllers to execute)
        5. Update node states
        """
        self.simulate_demand()
        self.simulate_production()
        self.get_controller_actions()
        self.simulate_utilities()
        self.update_nodes()
        self.time += self.time_step

    def simulate_demand(self):
        for component in self.demand_type_components:
            temp = component.step()

    def simulate_production(self):
        for component in self.producer_type_components:
            temp = component.step()

    def get_controller_actions(self):
        self.actions = []
        for controller in self.controllers:
            self.actions.append(controller.get_action())

    def simulate_utilities(self):
        for utility in self.utility_type_components:
            utility.step(self.actions[utility])

    def update_nodes(self):
        # accumulate heat per node in this step
        delta = {n: 0.0 for n in self.nodes}
        # Ask each component for heat contributions (explicit)
        for c in self.components:
            contribs = c.step(self.dt_s, self.nodes)
            # record component-level metric (e.g., total power)
            # here we store sum of absolute heat (J) as simple metric
            self.comp_history[c.name].append(sum(contribs.values())/self.dt_s if contribs else 0.0)
            for node_name, dd in contribs.items():
                if isinstance(self.nodes[node_name], DynamicNode):
                    delta[node_name] += dd / self.nodes[node_name].inertia
                elif isinstance(self.nodes[node_name], BalanceNode):
                    delta[node_name] += dd
        # update node state variables
        for name, node in self.dynamic_nodes.items():
            node.state_variable += delta[name]
            self.history[name].append(node.state_variable)
        for name, node in self.balance_nodes.items():
            if node.check_balance() == False:
                raise(NodeImbalanceError, f'Node imbalance error for node {name} at time step {self.time:.2f}')
        self.time.append(self.time)

    def run(self, t0_s: float, t_end_s: float):
        t = t0_s
        while t < t_end_s - 1e-9:
            self.step(t)
            t += self.dt_s

    def to_dataframe(self):
        df_nodes = pd.DataFrame(self.history, index=pd.Index(self.time, name='time_s'))
        df_comps = pd.DataFrame(self.comp_history, index=pd.Index(self.time, name='time_s'))
        return df_nodes, df_comps