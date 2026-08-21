from energy_system_control.components.base import StorageUnit
from energy_system_control.helpers import *
from energy_system_control.core.base_classes import InitContext
from energy_system_control.constants import WATER
from energy_system_control.sim.state import SimulationState
from typing import Dict, List
from scipy.linalg import solve_banded
import warnings, math

        

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
        self.heat_input_port_names = [self.main_heat_input_port_name, self.aux_heat_input_port_name]
        self.fluid_port_names = [self.cold_water_input_port_name, self.hot_water_output_port_name]
        
        super().__init__(name, {self.cold_water_input_port_name: 'fluid',
                                      self.hot_water_output_port_name: 'fluid',
                                      self.main_heat_input_port_name: 'heat',
                                      self.aux_heat_input_port_name: 'heat'})
    
    def step(self, state: SimulationState, action):
        # Heat port inputs are provided from external sources. Hot water output also. Hence, only the cold water input is updated
        self.ports[self.cold_water_input_port_name].flows['mass'] = -self.ports[self.hot_water_output_port_name].flows['mass']
        self.ports[self.cold_water_input_port_name].flows['heat'] = abs(self.ports[self.cold_water_input_port_name].flows['mass']) * WATER.cp * self.ports[self.cold_water_input_port_name].T
        heat_losses = self.calculate_losses(state)
        heat_input = 0.0
        for input_port in self.heat_input_port_names:
            if input_port in self.ports.keys():
                heat_input += self.ports[input_port].flows['heat']
        heat_fluid = self.ports[self.hot_water_output_port_name].flows['heat'] + self.ports[self.cold_water_input_port_name].flows['heat']
        self.temperature += (heat_input + heat_fluid + heat_losses) * state.time_step / (WATER.cp * self.volume * WATER.rho)
        self.SOC = self.temperature_to_SOC(state)

    def calculate_losses(self, state: SimulationState):
        ambient_temperature = self.T_amb if self.located_inside else state.environmental_data.temperature_ambient
        losses = -self.convection_coefficient_losses * self.surface * (self.temperature - ambient_temperature) * 1e-3
        return losses
    
    def set_inherited_fluid_port_values(self, state: SimulationState):
        # Allows for the temperature of the fluid leaving the tank to be set by the storage unit
        if self.hot_water_output_port_name in self.ports.keys():
            self.ports[self.hot_water_output_port_name].T = self.temperature
        return {self.hot_water_output_port_name: self.temperature}
    
    def set_inherited_heat_port_values(self, state: SimulationState):
        output = {}
        for port_name in self.heat_input_port_names:
            if port_name in self.ports.keys():
                self.ports[port_name].T = self.temperature
                output[port_name] = self.temperature
        return output
    
    def temperature_to_SOC(self, state: SimulationState):
        try:
            T_cold_water = state.environmental_data.temperature_cold_water
        except (AttributeError, KeyError):
            T_cold_water = C2K(20)
        return (self.temperature - T_cold_water) / (self.max_temperature - T_cold_water)

    def initialize(self, state: SimulationState):
        self.temperature = self.T_0
        self.SOC_0 = self.temperature_to_SOC(state)
        for port_name in self.heat_input_port_names:
            if port_name in self.ports.keys():
                self.ports[port_name].T = self.T_0
        super().initialize(state)


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
        tank_height: float, optional
            The height of the tank [m]. If not specificed, it is assumed a height-to-diameter ratio equal to 2.0
        height_cold_water_input [m]: float, opional
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
        self.heating_source_locations = {
            self.main_heat_input_port_name: self.main_heating_source_location, 
            self.aux_heat_input_port_name: self.aux_heating_source_location}
        self.matrix_B = None
        self.matrix_A = None
        self.water_mass_flow_t = None
    
    def identify_heat_input_layers(self, input_heights: float | list | None = None, default: int | None = None):
        vector_with_heat_input_layers = np.zeros(self.number_of_layers, dtype=np.float16)
        if not input_heights:
            default = default if default else 0
            vector_with_heat_input_layers[default] = 1
        elif isinstance(input_heights, (int, float)):
            vector_with_heat_input_layers[self.identify_layer_by_height(height = input_heights, default = default, output_type='layer_id')] = 1
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
        layer_id = int(self.number_of_layers - height // self.layer_height - 1) if height else default
        if output_type == 'vector':
            output = np.zeros(self.number_of_layers, dtype=np.int16)
            output[layer_id] = 1
            return output
        elif output_type == 'layer_id':
            return layer_id
    
    def step_backup(self, state: SimulationState, action):
        output = {}
        self._check_state(state)
        change_in_water_mass_flow = not math.isclose(self.water_mass_flow_t, -self.ports[self.hot_water_output_port_name].flows['mass'], abs_tol = 1e-4)
        self.water_mass_flow_t = -self.ports[self.hot_water_output_port_name].flows['mass']
        self.update_A_matrix(change_in_water_mass_flow)
        C = self.create_C_vector(state)
        D = -(self.matrix_B * self.T_layer + C)
        self.T_layer = solve_banded((1, 1), self.matrix_A, D)  
        self.temperature = self.T_layer.mean()
        self.SOC = self.temperature_to_SOC(state)
        # In the end, the only value that needs updating is the input from the cold water grid
        self.ports[self.cold_water_input_port_name].flows['mass'] = self.water_mass_flow_t
        self.ports[self.cold_water_input_port_name].flows['heat'] = self.water_mass_flow_t * WATER.cp * self.ports[self.cold_water_input_port_name].T
        return output

    def step(self, state: SimulationState, action):
        # Sanity check of the current state
        self._check_state(state)
        # Checking if water mass flows changed with respect to the previous time step
        update_coefficients = self._check_need_to_update_coefficients()
        internal_water_flows, inlet_water_flow, outlet_water_flow = self._update_water_flows()
        if update_coefficients is True:
            internal_heat_exchange_coefficients = self._update_heat_transfer_coefficients()
            self._update_A_matrix(internal_heat_exchange_coefficients, internal_water_flows, outlet_water_flow)
        C = self._create_C_vector(state, inlet_water_flow)
        D = -(self.matrix_B * self.T_layer + C)
        # Solve the thrediagonal system
        T_new = solve_banded((1, 1), self.matrix_A, D)
        # Assign the new temperature values to the layer temperatures
        self.T_layer = T_new
        # Calculate the overall average temperature and related SOC
        self.temperature = self.T_layer.mean()
        self.SOC = self.temperature_to_SOC(state)
        # In the end, the only value that needs updating is the input from the cold water grid
        self.ports[self.cold_water_input_port_name].flows['mass'] = self.water_mass_flow_t
        self.ports[self.cold_water_input_port_name].flows['heat'] = self.water_mass_flow_t * WATER.cp * self.ports[self.cold_water_input_port_name].T
        return {}

    def _check_need_to_update_coefficients(self):
        """
        Checks the need to update the heat transfer coefficients and, consequently, the matrix A. 
        Returns a boolean (True: matrix A needs updating. False: it does not)
        The decision is based on two checks:
        - If there was a change in the water mass flow (that is, for instance, if there was a hot water demand at t-1 and not at t)
        - If there is a change in the relative temperature of the layers (that is, if T_i(t-1) > T_i+1(t-1) and T_i(t) < T_i+1(t))
        """
        # Check for changes in water mass flows
        change_in_water_mass_flow = not math.isclose(self.water_mass_flow_t, -self.ports[self.hot_water_output_port_name].flows['mass'], abs_tol = 1e-4)
        self.water_mass_flow_t = -self.ports[self.hot_water_output_port_name].flows['mass']
        # Check for changes in relative temperature layers
        relative_temperature_layers_state_new = np.array([0] * (self.number_of_layers + 1), dtype=bool)
        relative_temperature_layers_state_new[1:-1] = self.T_layer[1:] > self.T_layer[:-1]
        change_in_relative_temperature_layers_state = any(relative_temperature_layers_state_new != self.relative_temperature_layers_state)
        # Finally calculating the decision on whether to update the matrix A
        output = change_in_relative_temperature_layers_state or change_in_water_mass_flow
        # Updating the relative temperature layers state
        self.relative_temperature_layers_state = relative_temperature_layers_state_new
        return output

    def _update_water_flows(self):
        """
        Determine water mass flows associated with the tank.

        Returns
        -------
        internal_flows : np.ndarray
            Mass flow between adjacent layers [kg/s].

        inlet_flow : float
            Mass flow entering the tank [kg/s].

        outlet_flow : float
            Mass flow leaving the tank [kg/s].
        """

        n = self.number_of_layers
        internal_flows = np.zeros(n - 1)
        # Retrieving the external water demand
        if self.ports[self.hot_water_output_port_name].flows['mass'] is not None:
            water_demand = -self.ports[self.hot_water_output_port_name].flows['mass']
        else:
            water_demand = 0.0
        # If it's zero, simplifying the calculation: It's all zeros
        if water_demand <= 0:
            return internal_flows, 0.0, 0.0
        # Retrieving inlet and outlet layers
        inlet = self.cold_water_input_location.argmax()  # Finds the layer where the array is equal to 1
        outlet = self.hot_water_output_location.argmax()  # same as above
        # Setting inlet and outlet flows
        inlet_flow = water_demand
        outlet_flow = water_demand
        # Checking relative node position
        if inlet > outlet:
            # Flow from lower layers toward upper layers
            internal_flows[outlet:inlet] = water_demand
        elif inlet < outlet:
            # Flow from upper layers toward lower layers
            internal_flows[inlet:outlet] = -water_demand

        return internal_flows, inlet_flow, outlet_flow

    def _update_heat_transfer_coefficients(self):
        # First we update the vector of internal heat exchange coefficients
        internal_heat_exchange_coefficients_new = np.ones(self.number_of_layers+1, dtype=np.float32) * WATER.k
        internal_heat_exchange_coefficients_new[0] = 0.0
        internal_heat_exchange_coefficients_new[self.number_of_layers] = 0.0
        internal_heat_exchange_coefficients_new[self.relative_temperature_layers_state==1] = WATER.k * self.convection_effect_coefficient
        return internal_heat_exchange_coefficients_new

    def _update_A_matrix(self, 
                         internal_heat_exchange_coefficients: np.ndarray, 
                         internal_water_flows: np.ndarray, 
                         outlet_water_flow: float):
        n = self.number_of_layers
        # ------------------------------------------------------------------
        # Heat contributions
        # ------------------------------------------------------------------
        # Alpha term
        alpha_heat = (-internal_heat_exchange_coefficients[1:-1] * self.surface_cross_section / self.layer_height * 1e-3)
        # Beta term
        beta_heat = (-self.matrix_B
            + self.surface_cross_section / self.layer_height * (internal_heat_exchange_coefficients[1:] + internal_heat_exchange_coefficients[:-1])* 1e-3
            + self.convection_coefficient_losses * self.surface_losses_layer_vec * 1e-3
        ) 
        # Gamma term
        gamma_heat = (-internal_heat_exchange_coefficients[1:-1] * self.surface_cross_section / self.layer_height * 1e-3)
        # ------------------------------------------------------------------
        # Water-flow contributions
        # ------------------------------------------------------------------
        water_cp = WATER.cp
        # Initialization
        beta_water = np.zeros(n)
        gamma_water = np.zeros(n - 1)
        alpha_water = np.zeros(n - 1)
        for i, flow in enumerate(internal_water_flows):
            if flow > 0:
                # Flow from layer i+1 -> layer i
                # - layer i receives water from layer i+1
                # - layer i+1 loses water
                beta_water[i + 1] += flow * water_cp
                gamma_water[i] -= flow * water_cp
            elif flow < 0:
                # Flow from layer i -> layer i+1
                flow_abs = -flow
                beta_water[i] += flow_abs * water_cp
                alpha_water[i] -= flow_abs * water_cp
        # ------------------------------------------------------------------
        # External outlet
        # ------------------------------------------------------------------
        if outlet_water_flow > 0:
            outlet_node = self.hot_water_output_location.argmax()
            # Water leaving the tank carries enthalpy corresponding
            # to the temperature of the outlet node.
            beta_water[outlet_node] += outlet_water_flow * water_cp
        # ------------------------------------------------------------------
        # Assemble A
        # ------------------------------------------------------------------
        A = np.zeros((3, n), dtype=np.float32)
        A[0, 1:] = gamma_heat + gamma_water
        A[1, :] = beta_heat + beta_water
        A[2, :-1] = alpha_heat + alpha_water
        self.matrix_A = A
        
    def _create_C_vector(self, state: SimulationState, inlet_water_flow: float):
        ambient_temperature = self.T_amb if self.located_inside else state.environmental_data.temperature_ambient
        total_heat_from_main_heating_source = self.ports[self.main_heat_input_port_name].flows['heat']
        if self.aux_heat_input_port_name in self.ports.keys():
            total_heat_from_aux_heating_source = self.ports[self.aux_heat_input_port_name].flows['heat']
        else:
            total_heat_from_aux_heating_source = 0.0
        # Calculating useful vectors
        vector_cold_water_input = self.cold_water_input_location * inlet_water_flow * WATER.cp * self.ports[self.cold_water_input_port_name].T
        vector_heat_from_main_heating_source = total_heat_from_main_heating_source / self.main_heating_source_location.sum() * self.main_heating_source_location
        vector_heat_from_aux_heating_source = total_heat_from_aux_heating_source / self.aux_heating_source_location.sum() * self.aux_heating_source_location
        # Finally calculating the C vector
        C = -self.convection_coefficient_losses * self.surface_losses_layer_vec * ambient_temperature * 1e-3 - vector_heat_from_main_heating_source - vector_heat_from_aux_heating_source - vector_cold_water_input
        return C
    
    def set_inherited_fluid_port_values(self, state):
        T_port = self.T_layer[np.nonzero(self.hot_water_output_location==1)][0]
        self.ports[self.hot_water_output_port_name].T = T_port
        return {self.hot_water_output_port_name: T_port}
    
    def set_inherited_heat_port_values(self, state):
        output = {}
        for port_name in self.heat_input_port_names:
            if port_name in self.ports.keys():
                T_heating_port = self.T_layer[self.heating_source_locations[port_name]==1].max()
                self.ports[port_name].T = T_heating_port
                output[port_name] = T_heating_port
        return output
        
    def initialize(self, ctx: InitContext):
        state = ctx.state
        self.water_mass_flow_t = 0.0
        self.T_layer = self.T_layer = np.array([self.T_0 - 0.01 * x for x in range(self.number_of_layers)], dtype=np.float32)
        self.relative_temperature_layers_state = np.zeros(self.number_of_layers + 1, dtype=np.int16)
        internal_heat_exchange_coefficients = self._update_heat_transfer_coefficients()
        internal_water_flows, inlet_water_flow, outlet_water_flow = self._update_water_flows()
        self.matrix_B = np.array([-self.layer_mass * WATER.cp / state.time_step] * self.number_of_layers, dtype=np.float32)
        self._update_A_matrix(internal_heat_exchange_coefficients, internal_water_flows, outlet_water_flow)
        super().initialize(ctx)

    def _check_state(self, state, stage="unknown"):
        """
        Function that makes a "sanity check" before each simulation step. 
        This helps in an early identification of potential issues
        """
        Tmin = C2K(0)
        Tmax = C2K(100)

        if not np.all(np.isfinite(self.T_layer)):
            raise RuntimeError(
                f"{self.name}: non-finite temperature at "
                f"t={state.time}, step={state.time_id}, stage={stage}\n"
                f"T_layer={self.T_layer}"
            )

        if np.any(self.T_layer < Tmin) or np.any(self.T_layer > Tmax):
            raise RuntimeError(
                f"{self.name}: physically unreasonable temperature at "
                f"t={state.time}, step={state.time_id}, stage={stage}\n"
                f"T_layer={self.T_layer}"
            )