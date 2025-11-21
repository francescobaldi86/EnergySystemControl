from energy_system_control.core.base_classes import Component
from energy_system_control.core.nodes import ThermalNode, MassNode, ElectricalStorageNode
from energy_system_control.helpers import *
from energy_system_control.constants import WATER
from typing import Dict, List
import warnings, math

class StorageUnit(Component):
    """A lumped hot-water store modeled as node with internal losses."""
    state: dict
    max_capacity: dict

    def __init__(self, name: str, nodes: List[str], max_capacity: Dict[str, float]):
        super().__init__(name, nodes)
        self.max_capacity = max_capacity

    def step(self, action):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        raise(NotImplementedError)
    
    def check_storage_state(self):
        # This function must be implemented for each sub type
        raise(NotImplementedError)
            
        

class HotWaterStorage(StorageUnit):
    tank_volume: float
    max_temperature: float
    def __init__(self, name, max_temperature: float, tank_volume: float, thermal_node: str | None = None, mass_node: str | None = None, T_0: float = 40.0):
        self.tank_volume = tank_volume
        self.max_temperature = C2K(max_temperature)
        self.T_0 = C2K(T_0)
        if thermal_node and mass_node:
            max_capacity = {
                thermal_node: max_temperature * tank_volume * 4.187,  # Energy capacity, in kJ
                mass_node: tank_volume
            }
            nodes = [thermal_node, mass_node]
        else:
            max_capacity = {
                f'{name}_thermal_node': max_temperature * tank_volume * 4.187,  # Energy capacity, in kJ
                f'{name}_mass_node': tank_volume
            }
            nodes = [f'{name}_thermal_node', f'{name}_mass_node']
        min_capacity = {n: 0.0 for n in max_capacity.keys()}
        super().__init__(name, nodes, max_capacity)

    def create_storage_nodes(self):
        nodes = {}
        nodes[f'{self.name}_thermal_node'] = ThermalNode(name = f'{self.name}_thermal_node', inertia = self.tank_volume * WATER.cp, T_0 = self.T_0)
        nodes[f'{self.name}_mass_node'] = MassNode(name = f'{self.name}_mass_node', max_capacity=self.tank_volume, m_0 = self.tank_volume)
        return nodes
    
    def step(self, action):
        output = {}
        for node_name, node in self.nodes.items():
            output[node_name] = 0.0
        return output


class MultiNodeHotWaterTank(StorageUnit):
    number_of_layers: int
    heat_injection_nodes: Dict[str, int]
    layer_cold_water_injection: int
    layer_hot_water_outlet: int
    T_0: float
    T_amb: float
    T_layer: np.array
    located_inside: bool
    conduction_coefficient_water: float
    convection_effect_coefficient: float
    convection_coefficient_losses: float
    volume: float
    height: float
    diameter: float
    layer_mass: float
    layer_height: float
    surface_cross_section: float
    surface_lateral_total: float
    surface_lateral_layer: float
    relative_temperature_layers_state: np.array
    internal_heat_exchange_coefficient: np.array
    ordered_layers: Dict[str, list]
    def __init__(self, name, volume: float, heat_injection_nodes: Dict[str,int], height: float | None = None, number_of_layers: int = 5, T_0: float = 40.0, convection_effect_coefficient: float = 1_000, convection_coefficient_losses: float = 0.8, located_inhouse: bool = True, T_amb: float = 22.0):
        """
        Model of hot water storage tank. Modeling reference is Leclercq et al. (2024) "Dynamic modeling and experimental validation of an electric water heater with a double storage tank configuraiton." ECOS 2024 Proceedings.
        Note that it is assumed that:
        - Layer numbering starts from the top: 0 is the top layer(node), N is the bottom layer (node)
        - Cold water injection happens at the bottom layer(node) only
        - Hot water extraction happens at top layer(node) only
        - Heat addition from each source can only happen within a single node (no serpentine heating multiple nodes)
        - Nodes cannot have multiple functions (e.g. the hot water outlet node cannot also be the heating node)

        Parameters
        ----------
        name : str
            Name of the component
        volume : float
            Storage capacity of the tank [l]
        heat_injection_nodes : Dict[str, int]
         	Dictionary containing one element for each node where heat is added. The key is the node name, the corresponding element represents the integer corresponding to the node where the heat is added
        height: float, optional
            The height of the tank. If not specificed, it is assumed a height-to-diameter ratio equal to 2.0
        number_of_layers: int, optional
            The number of layers(nodes) used in the tank module. Mininum value is 3, defaults to 5
        T_0: float, optional
            Starting temperature [°C] in the tank. Defaults to 40°C
        convection_effect_coefficient: float, optional
            The factor [-] by which the heat exchange coefficient between layers is increased when convection is included (when the temperature is higher in the lower layer). Defaults to 10_000, from Leclercq et al. (2024)
        convection_coefficient_losses: float, optional
            The convection coefficient [W/m2K] used to calculate losses to the ambient where the tank is located
        located_inside: bool, optional
            Defines the location where the tank is placed. If True the heat losses are calculated assuming T_amb. If False, the outer air temperature is used. Defaults to True
        T_amb: float, optional
            Temperature [°C] used to calculate heat losses to the ambient from the tank if located_inside is True. Defaults to 22.0
        """
        self.number_of_layers = number_of_layers
        self.heat_injection_nodes = heat_injection_nodes
        self.layer_cold_water_injection = self.number_of_layers - 1
        self.layer_hot_water_outlet = 0
        self.T_0 = C2K(T_0)
        self.located_inside = located_inhouse
        self.T_amb = C2K(T_amb)
        self.T_layer = np.array([self.T_0 - 0.01 * x for x in range(self.number_of_layers)])
        self.volume = volume * 1e-3  # Volume input is in liters, so it is converted to m3 to ensure the use of SI
        self.height = height if height else (volume * 16 / math.pi)**(1/3)  # based on the assumption of height over diameter equal to 2.0
        self.diameter = (4 * self.volume / self.height / math.pi)**0.5
        self.layer_mass = self.volume * WATER.rho / self.number_of_layers
        self.layer_height = self.height / self.number_of_layers
        self.surface_cross_section = math.pi * self.diameter**2 / 4
        self.surface_lateral_total = math.pi * self.diameter * self.height
        self.surface_lateral_layer = self.surface_lateral_total / self.number_of_layers
        self.relative_temperature_layers_state = np.zeros(self.number_of_layers-1)
        self.internal_heat_exchange_coefficient = np.ones(self.number_of_layers-1) * WATER.k
        self.ordered_layers = {'thermal': [], 'mass': []}  # List of thermal nodes, ordered from top to bottom. Top node is [0], bottom node is [N]
        self.convection_effect_coefficient = convection_effect_coefficient
        self.convection_coefficient_losses = convection_coefficient_losses
        super().__init__(name, [], {})
        

    def create_storage_nodes(self):
        nodes = {}
        for id in range(self.number_of_layers):
            if id == self.layer_cold_water_injection:
                thermal_node_name = f'{self.name}_cold_water_inlet_thermal_node'
                mass_node_name = f'{self.name}_cold_water_inlet_mass_node'
            elif id in [x for _, x in self.heat_injection_nodes.items()]:
                node_name = [name for name, key in self.heat_injection_nodes.items() if key == id][0]
                thermal_node_name = f'{self.name}_{node_name}_thermal_node'
                mass_node_name = f'{self.name}_{node_name}_mass_node'
            elif id == self.layer_hot_water_outlet:
                thermal_node_name = f'{self.name}_hot_water_outlet_thermal_node'
                mass_node_name = f'{self.name}_hot_water_outlet_mass_node'
            else:
                thermal_node_name = f'{self.name}_thermal_node_{id}'
                mass_node_name = f'{self.name}_mass_node_{id}'
            nodes[thermal_node_name] = ThermalNode(name = f'{self.name}_thermal_node', inertia = self.layer_mass * WATER.cp, T_0 = self.T_layer[id])
            nodes[mass_node_name] = MassNode(name = f'{self.name}_mass_node', max_capacity=self.layer_mass, m_0 = self.layer_mass)
            self.ordered_layers['thermal'].append(thermal_node_name)
            self.node_names.append(thermal_node_name)
            self.ordered_layers['mass'].append(mass_node_name)
            self.node_names.append(mass_node_name)
        return nodes 

    def step(self, action):
        output = {}
        ambient_temperature = self.T_amb if self.located_inside else self._environmental_data()['Temperature ambient']
        self.T_layer = np.array([self.nodes[node_name].T for node_name in self.ordered_layers['thermal']])  # Updates the numpy array with layer temperatures, for ease of access
        heat_exchange_between_layers = self.calculate_heat_exchange_between_layers()
        water_mass_flow = -self.nodes[f'{self.name}_hot_water_outlet_mass_node'].flow[list(self.nodes[f'{self.name}_hot_water_outlet_mass_node'].flow.keys())[0]] / self.time_step
        for node_id, node_name in enumerate(self.ordered_layers['thermal']):
            heat_addition = 0  # Only a place holder, the heat addition is managed by external components
            heat_to_layer_above = heat_exchange_between_layers[node_id-1] if node_id != 0 else 0
            heat_from_layer_below = heat_exchange_between_layers[node_id] if node_id != self.number_of_layers-1 else 0
            heat_to_environment = self.convection_coefficient_losses * self.surface_lateral_layer * (1 + self.surface_cross_section / self.surface_lateral_layer * (node_id%(self.number_of_layers-1) == 0)) * (self.T_layer[node_id] - ambient_temperature) * 1e-3
            heat_from_water_flowing_in = water_mass_flow * WATER.cp * self.T_layer[node_id+1] if node_id != self.number_of_layers-1 else 0
            heat_to_water_flowing_out = water_mass_flow * WATER.cp * self.T_layer[node_id] if node_id != 0 else 0
            output[node_name] = (heat_addition + heat_from_layer_below - heat_to_layer_above + heat_from_water_flowing_in - heat_to_water_flowing_out - heat_to_environment) * self.time_step
            # Calculating mass flows
            output[node_name.replace('thermal', 'mass')] = -water_mass_flow*self.time_step if node_name.replace('thermal', 'mass') == f'{self.name}_cold_water_inlet_mass_node' else (water_mass_flow*self.time_step if node_name.replace('thermal', 'mass') == f'{self.name}_hot_water_outlet_mass_node' else 0)
        return output


    def calculate_heat_exchange_between_layers(self):
        # Method to calculate the internal heat exchange between layers. 
        # The output is a numpy array where each element represents the heat exchanged across the i-th interface. 
        # Calculate the relationship between the temperatures across different layers. Each element is 1 if the temperature below the interface is higher than the temperature above the interface
        relative_temperature_layers_state = self.T_layer[1:] > self.T_layer[:-1]
        # Compare the relative temperature layers state with the existing one. Only recalculate the internal heat exchange coefficient vector if changes happen
        if any(relative_temperature_layers_state != self.relative_temperature_layers_state):
            self.relative_temperature_layers_state = relative_temperature_layers_state
            self.internal_heat_exchange_coefficient = np.ones(self.number_of_layers-1) * WATER.k
            self.internal_heat_exchange_coefficient[self.relative_temperature_layers_state] = WATER.k * self.convection_effect_coefficient
        heat_exchange_between_layers = self.internal_heat_exchange_coefficient * self.surface_cross_section * (self.T_layer[1:] - self.T_layer[:-1]) * 1e-3  # [W/m2K] * [m2] * [K] * [kW/W] --> kW
        return heat_exchange_between_layers



class Battery(StorageUnit):
    crate: float
    erate: float
    efficiency_charge: float
    efficiency_discharge: float
    def __init__(self, name, capacity: float, electrical_node: str | None = None, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5):
        """
        Model of a simple electric battery

        Parameters
        ----------
        name : str
            Name of the component
        capacity : float
         	Design maximum energy capacity [kWh] of the battery.
        electrical_node : float, optional
           	Name of the electrical node of the battery. If not provided, it will automatically be assigned to f"{name}_electrical_node"
        crate : float, optional
            C-rate [-] of the battery. Sets the maximum charge power based on the capacity (if crate = 1, P_charge_max [kW] = E_max [kWh]) Defaults to 1.
        erate : float, optional
            E-rate [-] of the battery. Sets the maximum discharge power based on the capacity (if erate = 1, P_discharge_max [kW] = E_max [kWh]) Defaults to 1.
        efficiency_charge : float, optional
            Efficiency [-] of the charging process of the battery. Defaults to 0.92 [REF]
        efficiency_discharge : float, optional
            Efficiency [-] of the discharging process of the battery. Defaults to 0.94 [REF]
        SOC_0 : float, optional
            Battery state of charge [-] at simulation start. Defaults to 0.5
        """
        self.crate = crate
        self.erate = erate
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.SOC_0 = SOC_0
        self.max_battery_capacity = capacity * 3600  # Energy capacity, in kJ
        self.max_charging_power = capacity * self.crate
        self.max_discharging_power = capacity * self.erate
        self.io_node_name = electrical_node
        self.internal_node_name = f'{name}_electrical_node'
        super().__init__(name, [self.internal_node_name, self.io_node_name], {self.internal_node_name: self.max_battery_capacity})
    
    @property
    def SOC(self): 
        return self.nodes[self.node_names[0]].SOC

    def create_storage_nodes(self):
        nodes = {}
        nodes[f'{self.name}_electrical_node'] = ElectricalStorageNode(name = self.internal_node_name, max_capacity = self.max_capacity[self.internal_node_name], SOC_0 = self.SOC_0)
        # self.verify_connected_components()
        return nodes
    
    def check_storage_state(self):
        if self.SOC > 1.0:
            warnings.warn(f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.time}. Observed SOC is {self.SOC} while max value is 1.0', UserWarning)
        elif self.SOC < 0.0:
            warnings.warn(f'Storage unit {self.name} has storage level lower than minimum allowed at time step {self.time}. Observed SOC is {self.SOC} while min value is 0.0', UserWarning)
    
    def verify_connected_components(self):
        raise(NotImplementedError)
    
    def step(self, action):
        # Just considering charging/discharging losses
        self.check_storage_state()
        if action > 0: 
            return {self.internal_node_name: action * self.efficiency_charge,
                    self.io_node_name: -action}
        else:
            return {self.internal_node_name: action * self.efficiency_discharge,
                    self.io_node_name: -action}
        
    def get_maximum_charge_power(self):
        raise NotImplementedError
    
    def get_maximum_discharge_power(self):
        raise NotImplementedError
    

class LithiumIonBattery(Battery):
    SOC_min: float
    SOC_max: float
    def __init__(self, name, capacity: float, electrical_node: str | None = None, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5, SOC_min: float = 0.3, SOC_max: float = 0.9):
        super().__init__(name, capacity, electrical_node, crate, erate, efficiency_charge, efficiency_discharge, SOC_0)
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max

    def get_maximum_charge_power(self):
        # In a lithium-ion battery, we assume that the maximum charging power is constant between SOC_min and SOC_max, and then decreases linearly to zero between SOC_max and 1.0
        if self.SOC < self.SOC_max:
            return self.max_charging_power
        else:
            return self.max_charging_power * (1.0 - self.SOC) / (1.0 - self.SOC_max)
    
    def get_maximum_discharge_power(self):
        # In a lithium-ion battery, we assume that the maximum discharging power is constant between SOC_min and SOC_max, and then decreases linearly to zero between SOC_min and 0.0
        if self.SOC > self.SOC_min:
            return self.max_discharging_power
        else:
            return self.max_discharging_power * self.SOC / self.SOC_min