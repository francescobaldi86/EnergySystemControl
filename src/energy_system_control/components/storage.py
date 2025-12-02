from energy_system_control.components.base import Component
from energy_system_control.helpers import *
from energy_system_control.constants import WATER
from typing import Dict, List
from scipy.linalg import solve_banded
import warnings, math


class StorageUnit(Component):
    """Generic storage unit"""
    max_capacity: float
    SOC: float

    def __init__(self, name: str, ports_info: dict):
        super().__init__(name, ports_info)

    def step(self, action):
        # Storage doesn't actively add Q (unless charged/discharged), but could implement losses
        raise(NotImplementedError)
    
    def calculate_losses(self):
        raise(NotImplementedError)
    
    def check_storage_state(self):
        # This function must be implemented for each sub type
        raise(NotImplementedError)
    
    def reset(self):
        self.SOC = self.SOC_0
            
        

class HotWaterStorage(StorageUnit):
    volume: float
    surface: float
    height: float
    diameter: float
    max_temperature: float
    cold_water_input_port_name: str
    hot_water_output_port_name: str
    main_heat_input_port_name: str
    aux_heat_input_port_name: str
    temperature: float
    convection_coefficient_losses: float

    def __init__(self, 
                 name, 
                 tank_volume: float, 
                 tank_height: float|None = None,
                 max_temperature: float = 80,
                 T_0: float = 40.0, 
                 convection_coefficient_losses: float = 0.8, 
                 located_inside: bool = True, 
                 T_amb: float = 22):
        """
        Simplified model of hot water storage tank, assuming perfect mixing.
        Has potentially two heat sources: main and auxiliary

        Parameters
        ----------
        name : str
            Name of the component
        volume : float
            Storage capacity of the tank [l]
        height: float, optional
            The height of the tank. If not specificed, it is assumed a height-to-diameter ratio equal to 2.0
        T_0: float, optional
            Starting temperature [°C] in the tank. Defaults to 40°C
        convection_coefficient_losses: float, optional
            The convection coefficient [W/m2K] used to calculate losses to the ambient where the tank is located
        located_inside: bool, optional
            Defines the location where the tank is placed. If True the heat losses are calculated assuming T_amb. If False, the outer air temperature is used. Defaults to True
        T_amb: float, optional
            Temperature [°C] used to calculate heat losses to the ambient from the tank if located_inside is True. Defaults to 22.0
        """
        self.volume = tank_volume * 1e-3  # Volume input is in liters, so it is converted to m3 to ensure the use of SI
        self.height = tank_height if tank_height else (self.volume * 16 / math.pi)**(1/3)  # based on the assumption of height over diameter equal to 2.0
        self.diameter = (4 * self.volume / self.height / math.pi)**0.5
        self.surface = math.pi * self.diameter * (self.height + 0.5 * self.diameter) 
        self.max_temperature = C2K(max_temperature)
        self.T_0 = C2K(T_0)
        self.T_amb = C2K(T_amb)
        self.convection_coefficient_losses = convection_coefficient_losses
        self.located_inside = located_inside
        self.cold_water_input_port_name = f'{name}_cold_water_input_port'
        self.hot_water_output_port_name = f'{name}_hot_water_output_port'
        self.main_heat_input_port_name = f'{name}_main_heat_input_port'
        self.aux_heat_input_port_name = f'{name}_aux_heat_input_port'
        
        super().__init__(name, {self.cold_water_input_port_name: 'fluid',
                                      self.hot_water_output_port_name: 'fluid',
                                      self.main_heat_input_port_name: 'heat',
                                      self.aux_heat_input_port_name: 'heat'})
    
    def step(self, action):
        # Heat port inputs are provided from external sources. Hot water output also. Hence, only the cold water input is updated
        self.ports[self.cold_water_input_port_name].flow['mass'] = -self.ports[self.hot_water_output_port_name].flow['mass']
        self.ports[self.cold_water_input_port_name].flow['heat'] = abs(self.ports[self.cold_water_input_port_name].flow['mass']) * WATER.cp * self.ports[self.cold_water_input_port_name].T
        heat_losses = self.calculate_losses()
        heat_input = self.ports[self.main_heat_input_port_name].flow['heat'] + self.ports[self.aux_heat_input_port_name].flow['heat'] 
        heat_fluid = self.ports[self.hot_water_output_port_name].flow['heat'] + self.ports[self.cold_water_input_port_name].flow['heat']
        self.temperature += (heat_input + heat_fluid + heat_losses) / (WATER.cp * self.volume * WATER.rho)
        self.SOC = self.temperature_to_SOC()

    def calculate_losses(self):
        ambient_temperature = self.T_amb if self.located_inside else self._environmental_data()['Temperature ambient']
        losses = -self.convection_coefficient_losses * self.surface * (self.temperature - ambient_temperature) * 1e-3
        return losses
    
    def set_inherited_fluid_port_values(self):
        self.ports[self.hot_water_output_port_name].T = self.temperature
        return self.hot_water_output_port_name, self.temperature
    
    def set_inherited_heat_port_values(self):
        self.ports[self.main_heat_input_port_name].T = self.temperature
        return self.main_heat_input_port_name, self.temperature
    
    def temperature_to_SOC(self):
        try:
            T_cold_water = self._environmental_data()['Temperature cold water']
        except (AttributeError, KeyError):
            T_cold_water = C2K(20)
        return (self.temperature - T_cold_water) / (self.max_temperature - T_cold_water)

    def reset(self):
        self.temperature = self.T_0
        self.SOC_0 = self.temperature_to_SOC()
        super().reset()


class MultiNodeHotWaterTank(HotWaterStorage):
    number_of_layers: int
    heat_injection_nodes: Dict[str, int]
    layer_cold_water_injection: int
    layer_hot_water_outlet: int
    T_layer: np.array
    convection_effect_coefficient: float
    layer_mass: float
    layer_height: float
    surface_cross_section: float
    surface_lateral_layer: float
    relative_temperature_layers_state: np.array
    internal_heat_exchange_coefficient: np.array
    ordered_layers: Dict[str, list]
    matrix_A: np.array
    matrix_B: np.array

    def __init__(self, 
                 name, 
                 tank_volume: float, 
                 tank_height: float | None = None, 
                 max_temperature: float = 80,
                 height_cold_water_input: float | None = None,
                 height_hot_water_output: float | None = None,
                 height_main_heat_input: float | list | None = None,
                 height_aux_heat_input: float | list | None = None, 
                 number_of_layers: int = 5, 
                 T_0: float = 40.0, 
                 convection_effect_coefficient: float = 1_000, 
                 convection_coefficient_losses: float = 0.8, 
                 located_inside: bool = True, 
                 T_amb: float = 22.0):
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
        height_cold_water_input: float, opional
            The height at which cold water is injected in the tank. By default it is assumed that this happens in the bottom layer
        number_of_layers: int, optional
            The number of layers(nodes) used in the tank module. Mininum value is 3, defaults to 5
        T_0: float, optional
            Starting temperature [°C] in the tank. Defaults to 40°C
        convection_effect_coefficient: float, optional
            The factor [-] by which the heat exchange coefficient between layers is increased when convection is included (when the temperature is higher in the lower layer). Defaults to 10_000, from Leclercq et al. (2024)
        convection_coefficient_losses: float, optional
            The convection coefficient [W/m2K] used to calculate losses to the ambient where the tank is located. Defaults to 0.8
        located_inside: bool, optional
            Defines the location where the tank is placed. If True the heat losses are calculated assuming T_amb. If False, the outer air temperature is used. Defaults to True
        T_amb: float, optional
            Temperature [°C] used to calculate heat losses to the ambient from the tank if located_inside is True. Defaults to 22.0
        """
        super().__init__(name, 
                         tank_volume = tank_volume, 
                         tank_height = tank_height, 
                         max_temperature = max_temperature,
                         T_0 = T_0, 
                         convection_coefficient_losses = convection_coefficient_losses, 
                         located_inside = located_inside, 
                         T_amb = T_amb)
        self.number_of_layers = number_of_layers
        self.layer_mass = self.volume * WATER.rho / self.number_of_layers
        self.layer_height = self.height / self.number_of_layers
        self.surface_cross_section = math.pi * self.diameter**2 / 4
        self.surface_lateral_total = math.pi * self.diameter * self.height
        self.surface_lateral_layer = self.surface_lateral_total / self.number_of_layers
        self.surface_losses_layer_vec = np.ones(self.number_of_layers) * self.surface_lateral_layer
        self.surface_losses_layer_vec[0] += self.surface_cross_section
        self.surface_losses_layer_vec[self.number_of_layers-1] += self.surface_cross_section
        self.convection_effect_coefficient = convection_effect_coefficient
        self.convection_coefficient_losses = convection_coefficient_losses
        # Identifying layers for heat exchange with the ports
        self.cold_water_input_location = self.identify_layer_by_height(height = height_cold_water_input, default = self.number_of_layers-1)
        self.hot_water_output_location = self.identify_layer_by_height(height = height_hot_water_output, default = 0)
        self.main_heating_source_location = self.identify_heat_input_layers(height_main_heat_input, default=self.number_of_layers-1)
        self.aux_heating_source_location = self.identify_heat_input_layers(height_aux_heat_input, default=self.number_of_layers-1)
        self.matrix_B = None
        self.matrix_A = None
    
    def identify_heat_input_layers(self, input_heights: float | list | None = None, default: int | None = None):
        vector_with_heat_input_layers = np.zeros(self.number_of_layers, dtype=np.float16)
        if not input_heights:
            default = default if default else 0
            vector_with_heat_input_layers[default] = 1
        elif isinstance(input_heights, (int, float)):
            vector_with_heat_input_layers[self.identify_layer_by_height(height = input_heights, default = default)] = 1
        elif isinstance(input_heights, list):
            if len(input_heights) != 2:
                raise(IndexError, f'The length of the heat input heights of the hot water storage {self.name} should be provided either as a float or as a list with two elements')
            else:
                for layer in range(self.number_of_layers):
                    layer_start_height = self.height - (layer + 1) * self.layer_height
                    layer_end_height = self.height - layer * self.layer_height
                    if (layer_start_height >= input_heights[0]) and (layer_end_height <= input_heights[1]):
                        vector_with_heat_input_layers[layer] = 1.0
                    elif (layer_start_height < input_heights[0]) and (layer_end_height <= input_heights[1]):
                        vector_with_heat_input_layers[layer] = (layer_end_height - input_heights[0]) / self.layer_height
                    elif (layer_start_height >= input_heights[0]) and (layer_end_height > input_heights[1]):
                        vector_with_heat_input_layers[layer] = (input_heights[1] - layer_start_height) / self.layer_height
                    else:
                        vector_with_heat_input_layers[layer] = 0
        return vector_with_heat_input_layers

    def identify_layer_by_height(self, height: float|int|None, default: int, output_type: str = 'vector'):
        layer_id = self.number_of_layers - height // self.layer_height - 1 if height else default
        if output_type == 'vector':
            output = np.zeros(self.number_of_layers, dtype=np.int16)
            output[layer_id] = 1
            return output
        elif output_type == 'layer_id':
            return layer_id
    
    def step(self, action):
        output = {}
        change_in_water_mass_flow = not math.isclose(self.water_mass_flow_t, -self.ports[self.hot_water_output_port_name].flow['mass'] / self.time_step, abs_tol = 1e-4)
        self.water_mass_flow_t = -self.ports[self.hot_water_output_port_name].flow['mass'] / self.time_step
        self.update_A_matrix(change_in_water_mass_flow)
        C = self.create_C_vector()
        D = -(self.matrix_B * self.T_layer + C)
        self.T_layer = solve_banded((1, 1), self.matrix_A, D)  
        self.temperature = self.T_layer.mean()
        self.SOC = self.temperature_to_SOC()
        # In the end, the only value that needs updating is the input from the cold water grid
        self.ports[self.cold_water_input_port_name].flow['mass'] = self.water_mass_flow_t * self.time_step
        self.ports[self.cold_water_input_port_name].flow['heat'] = self.water_mass_flow_t * WATER.cp * self.ports[self.cold_water_input_port_name].T
        return output

    def update_A_matrix(self, change_in_water_mass_flow):
        # First we check if the matrix need updating
        relative_temperature_layers_state = np.array([0] * (self.number_of_layers + 1), dtype=bool)
        relative_temperature_layers_state[1:-1] = self.T_layer[1:] > self.T_layer[:-1]
        # Compare the relative temperature layers state with the existing one. Only recalculate the internal heat exchange coefficient vector if changes happen
        if any(relative_temperature_layers_state != self.relative_temperature_layers_state) or change_in_water_mass_flow:
            self.relative_temperature_layers_state = relative_temperature_layers_state
            # First we update the vector of internal heat exchange coefficients
            internal_heat_exchange_coefficient = np.ones(self.number_of_layers+1, dtype=np.float32) * WATER.k
            internal_heat_exchange_coefficient[0] = 0.0
            internal_heat_exchange_coefficient[self.number_of_layers] = 0.0
            internal_heat_exchange_coefficient[self.relative_temperature_layers_state] = WATER.k * self.convection_effect_coefficient
            # Calculating diagonals
            alpha = - internal_heat_exchange_coefficient[1:-1] * self.surface_cross_section / self.layer_height * 1e-3  # Power values are converted to kW
            beta = -self.matrix_B + self.water_mass_flow_t * WATER.cp + self.surface_cross_section / self.layer_height * (internal_heat_exchange_coefficient[1:]+internal_heat_exchange_coefficient[:-1]) * 1e-3 + self.convection_coefficient_losses * self.surface_losses_layer_vec * 1e-3
            gamma = - self.water_mass_flow_t * WATER.cp - internal_heat_exchange_coefficient[1:-1] * self.surface_cross_section / self.layer_height * 1e-3
            # Finally creating the A matrix
            A = np.zeros((3, self.number_of_layers), dtype=np.float32)
            A[0, 1:] = gamma
            A[1, :] = beta
            A[2, :-1] = alpha
            self.matrix_A = A
        
    def create_C_vector(self):
        ambient_temperature = self.T_amb if self.located_inside else self._environmental_data()['Temperature ambient']
        total_heat_from_main_heating_source = self.ports[self.main_heat_input_port_name].flow['heat'] / self.time_step
        total_heat_from_aux_heating_source = self.ports[self.aux_heat_input_port_name].flow['heat'] / self.time_step
        # Calculating useful vectors
        vector_cold_water_input = self.cold_water_input_location * self.water_mass_flow_t * WATER.cp * self.ports[self.cold_water_input_port_name].T
        vector_heat_from_main_heating_source = total_heat_from_main_heating_source / len([self.main_heating_source_location]) * self.main_heating_source_location
        vector_heat_from_aux_heating_source = total_heat_from_aux_heating_source / len([self.aux_heating_source_location]) * self.aux_heating_source_location
        # Finally calculating the C vector
        C = -self.convection_coefficient_losses * self.surface_losses_layer_vec * ambient_temperature * 1e-3 - vector_heat_from_main_heating_source - vector_heat_from_aux_heating_source - vector_cold_water_input
        return C

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
    
    def set_inherited_fluid_port_values(self):
        T_port = self.T_layer[np.nonzero(self.hot_water_output_location==1)]
        self.ports[self.hot_water_output_port_name].T = T_port
        return self.hot_water_output_port_name, T_port
    
    def set_inherited_heat_port_values(self):
        T_heating_port = self.T_layer[self.main_heating_source_location==1].max()
        self.ports[self.main_heat_input_port_name].T = T_heating_port
        return self.main_heat_input_port_name, T_heating_port
        
    def reset(self):
        self.water_mass_flow_t = 0.0
        self.T_layer = self.T_layer = np.array([self.T_0 - 0.01 * x for x in range(self.number_of_layers)], dtype=np.float32)
        self.relative_temperature_layers_state = np.zeros(self.number_of_layers + 1, dtype=np.int16)
        self.internal_heat_exchange_coefficient = np.ones(self.number_of_layers-1, dtype=np.float32) * WATER.k
        self.matrix_B = np.array([-self.layer_mass * WATER.cp / self.time_step] * self.number_of_layers, dtype=np.float32)
        self.update_A_matrix(True)
        super().reset()


class Battery(StorageUnit):
    crate: float
    erate: float
    efficiency_charge: float
    efficiency_discharge: float
    port_name: str
    max_charging_power: float
    max_discharging_power:float
    self_discharge_rate: float
    def __init__(self, name, capacity: float, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5, self_discharge_rate: float = 0.0):
        """
        Model of a simple electric battery

        Parameters
        ----------
        name : str
            Name of the component
        capacity : float
         	Design maximum energy capacity [kWh] of the battery.
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
        self_discharge_rate: float, optional
            Battery self discharge rate [-/h], indicating the fraction of the current charge lost per hour. Defaults to 0.0 (no self-discharge)
        """
        self.crate = crate
        self.erate = erate
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.SOC_0 = SOC_0
        self.max_capacity = capacity * 3600  # Energy capacity, in kJ
        self.max_charging_power = capacity * self.crate
        self.max_discharging_power = capacity * self.erate
        self.self_discharge_rate = self_discharge_rate / 3600  # Provided
        self.port_name = f'{name}_electricity_port'
        super().__init__(name, {self.port_name: 'electricity'})
    
    def check_storage_state(self):
        if self.SOC > 1.0:
            raise StorageError(f'Storage unit {self.name} has storage level higher than maximum allowed at time step {self.time}. Observed SOC is {self.SOC} while max value is 1.0')
        elif self.SOC < 0.0:
            raise StorageError(f'Storage unit {self.name} has storage level lower than minimum allowed at time step {self.time}. Observed SOC is {self.SOC} while min value is 0.0')
    
    def verify_connected_components(self):
        raise(NotImplementedError)
    
    def step(self, action):
        # Just considering charging/discharging losses
        self.check_storage_state()
        if abs(self.ports[self.port_name].flow['electricity']) > 100:
            pass
        if self.ports[self.port_name].flow['electricity'] > 0:  # Charging
            self.SOC += self.ports[self.port_name].flow['electricity'] * self.efficiency_charge / self.max_capacity
        else:
            self.SOC += self.ports[self.port_name].flow['electricity'] / self.efficiency_discharge / self.max_capacity
        self.SOC -= self.SOC * self.self_discharge_rate * self.max_capacity * self.time_step / 3600 # The self discharge is input in fraction of current capacity per hour
        
    def get_maximum_charge_power(self):
        return self.max_charging_power
    
    def get_maximum_discharge_power(self):
        return self.max_discharging_power
    

class LithiumIonBattery(Battery):
    SOC_min: float
    SOC_max: float
    def __init__(self, name, capacity: float, crate: float = 1.0, erate: float = 1.0, efficiency_charge: float = 0.92, efficiency_discharge: float = 0.94, SOC_0: float = 0.5, SOC_min: float = 0.3, SOC_max: float = 0.9, self_discharge_rate: float = 0.025/30/24):
        super().__init__(name, capacity, crate, erate, efficiency_charge, efficiency_discharge, SOC_0, self_discharge_rate)
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