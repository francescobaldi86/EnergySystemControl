from typing import Dict, List
from collections import defaultdict
import pandas as pd
from EnergySystemControl.environments.base_classes import *
from EnergySystemControl.environments.demands import *
from EnergySystemControl.environments.producers import *
from EnergySystemControl.environments.utilities import *
from EnergySystemControl.environments.storage_units import *
from EnergySystemControl.environments.nodes import *
from EnergySystemControl.controllers.base_controller import *
from EnergySystemControl.helpers import *



class Environment:
    def __init__(self, nodes: Dict[str, Node], components: List[Component], controllers: List[Controller]):
        self.nodes = nodes
        self.balance_nodes = {}
        self.dynamic_nodes = {}
        self.components = components
        self.components_classified = defaultdict(list)
        self.controllers: List[Controller] = controllers
        # Ordering data
        self.classify_components()
        self.create_storage_nodes()
        self.classify_nodes()

    def add_component(self, component_name, component_type, **kwargs):
        if component_type not in Component.registry:
            raise ValueError(f"Unknown component type: {component_type}")
        cls = Component.registry[component_type]
        self.components[component_name] = cls(**kwargs)
        

    def classify_nodes(self):
        # Classify nodes based on whether they are balance or dynamic nodes
        for name, node in self.nodes.items():
            if isinstance(node, BalanceNode):
                self.balance_nodes[name] = node
            elif isinstance(node, DynamicNode):
                self.dynamic_nodes[name] = node

    def classify_components(self):
        # Classify components based on their type
        for name, component in self.components.items():
            if isinstance(component, Demand):
                self.components_classified['Demand'].append(component)
            elif isinstance(component, Producer):
                self.components_classified['Producer'].append(component)
            elif isinstance(component, BalancingUtility):
                self.components_classified['BalancingUtility'].append(component)
            elif isinstance(component, Utility):
                self.components_classified['Utility'].append(component)
            elif isinstance(component, StorageUnit):
                self.components_classified['StorageUnit'].append(component)

    def create_storage_nodes(self):
        for component in self.components_classified['StorageUnit']:
            self.nodes.update(component.create_storage_nodes())

    def read_timeseries_data(self):
        for _, component in self.components.items():
            if callable(getattr(component, 'resample_data', None)):
                component.resample_data(self.time_step, self.time_end)


    def step(self):
        """
        Taking a step involves the four main actions, executed in sequence:
        1. Simulate "demand" type components (they do not depend on other components' behaviour)
        2. Simulate "production" type components (they do not depend on other components' behaviour)
        3. Activate controllers
        4. Simulate "utility" type components (they require the respective controllers to execute)
        5. Update node states
        """
        # Put to 0 the delta of the balance at each node
        for _, node in self.nodes.items():
            node.delta = 0.0
        # Initialize control actions
        self.control_actions = defaultdict(lambda: None)
        self.components_to_simulate = [x for x in self.components.keys()]
        # Simulate components
        self.update_environmental_data()
        self.simulate_components_of_type('Demand')
        self.simulate_components_of_type('Producer')
        self.get_controller_actions()
        self.simulate_components_of_type('Utility')
        self.simulate_components_of_type('StorageUnit')
        self.simulate_components_of_type('BalancingUtility')
        if self.components_to_simulate != []:
            raise(BaseException, f'The step was concluded but components {self.components_to_simulate} where not simulated at time {self.time}, time ID {self.time_id}. Check what happened!')
        # update node state variables
        for name, node in self.dynamic_nodes.items():
            node.state_variable += node.delta
            self.history[name].append(node.state_variable)
        for name, node in self.balance_nodes.items():
            if node.check_balance() == False:
                raise(NodeImbalanceError, f'Node imbalance error for node {name} at time step {self.time:.2f}')

    def update_environmental_data(self):
        self.environmental_data = {
            'Temperature cold water': C2K(15),
            'Temperature ambient': C2K(20)
        }

    def simulate_components_of_type(self, type: str):
        components = self.components_classified[type]
        for component in components:
            if component.name in self.components_to_simulate:
                component.time = self.time
                component.time_id = self.time_id
                self.take_component_step(component, None)           
                self.components_to_simulate.remove(component.name)

    def get_controller_actions(self):
        for controller in self.controllers:
            controller.time = self.time
            controller.time_id = self.time_id
            obs = controller.get_obs(self)
            actions = controller.get_action(obs)
            for component_name, action in actions.items():
                if component_name in self.components_to_simulate:
                    self.take_component_step(self.components[component_name], action)
                    self.components_to_simulate.remove(component_name)
                else:
                    raise(KeyError, f'Component {component_name} has been simulated before its related control action was calculated at time {self.time}, time ID {self.time_id}. Check what happens!')
            self.control_actions.update(actions)

    def take_component_step(self, component, action):
        contribs = component.step(self.time_step, self.nodes, self.environmental_data, action)
        # Update node delta
        for node_name, dd in contribs.items():
            if isinstance(self.nodes[node_name], DynamicNode):
                self.nodes[node_name].delta += dd / self.nodes[node_name].inertia
            elif isinstance(self.nodes[node_name], BalanceNode):
                self.nodes[node_name].delta += dd
        # Update component history
        for node in component.nodes:
            self.comp_history[component.name][node].append(contribs[node] / self.time_step)  # Energy flows are saved in kW. Mass flows in kg/s      

    def run(self, time_start: float = 0.0, time_end: float = 8760.0, time_step: float = 0.5):
        self.time_step = time_step * 3600  # Time step is stored in seconds
        # Read environmental data
        self.environmental_data = {}
        # Data collection
        self.history = {n: [] for n in self.dynamic_nodes.keys()}
        self.comp_history = {c_name: {n: [] for n in c.nodes} for c_name, c in self.components.items()}
        # Time
        self.time_start = time_start * 3600  # Time values are stored in seconds
        self.time_end = time_end * 3600  # Time values are stored in seconds
        self.time = time_start
        self.time_id = 0
        self.time_vector = np.arange(self.time_start, self.time_end, self.time_step)
        self.read_timeseries_data()
        while self.time < self.time_end - 1e-9:
            self.step()
            self.time += self.time_step
            self.time_id += 1

    def to_dataframe(self):
        df_nodes = pd.DataFrame(self.history, index=pd.Index(self.time_vector / 3600, name='time_s'))
        df_comps = pd.concat(
            {comp: pd.DataFrame(nodes, index=pd.Index(self.time_vector / 3600, name='time_s')) for comp, nodes in self.comp_history.items()},
            axis=1)
        

        return df_nodes, df_comps