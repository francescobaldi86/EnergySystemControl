from energy_system_control.constants import WATER
from energy_system_control.components.base import ImplicitComponent
from energy_system_control.core.base_classes import InitContext
from energy_system_control.helpers import *
from energy_system_control.sim.state import SimulationState
from energy_system_control.components.storage_units.thermal_storage import HotWaterStorage, MultiNodeHotWaterTank
import numpy as np
import pandas as pd
from typing import Literal
from math import log

HeatExchangerType = Literal['parallel flow', 'counter-current flow', 'cross flow']

class HeatExchangerTwoFluids(ImplicitComponent):
    """
    Fluid-to-fluid heat exchanger model using the epsilon-NTU method.
    
    This component models a heat exchanger that transfers thermal energy between two fluid streams.
    The model uses the epsilon-NTU (effectiveness-Number of Transfer Units) method to calculate
    the heat transfer rate and outlet temperatures for both fluids. It supports three flow
    configurations: parallel flow, counter-current flow, and cross flow.
    
    The epsilon-NTU method is particularly useful for cases where the outlet temperatures are
    not known in advance and need to be calculated based on the effectiveness of the heat exchanger.
    
    Attributes
    ----------
    fluid_1_input_port_name : str
        Name of the input port for fluid 1
    fluid_1_output_port_name : str
        Name of the output port for fluid 1
    fluid_2_input_port_name : str
        Name of the input port for fluid 2
    fluid_2_output_port_name : str
        Name of the output port for fluid 2
    A : float
        Heat exchange surface area (m²)
    U : float
        Overall heat transfer coefficient (W/(m²·K))
    heat_exchanger_type : HeatExchangerType
        Configuration of the heat exchanger flow ('parallel flow', 'counter-current flow', or 'cross flow')
    """

    def __init__(self, name, exchange_surface: float, heat_exchange_coefficient: float, heat_exchanger_type: HeatExchangerType = 'counter-current flow'):
        """
        Initialize a heat exchanger component.
        
        Parameters
        ----------
        name : str
            Unique identifier for the heat exchanger
        exchange_surface : float
            Heat exchange surface area in m²
        heat_exchange_coefficient : float
            Overall heat transfer coefficient in W/(m²·K)
        heat_exchanger_type : HeatExchangerType, optional
            Flow configuration of the heat exchanger. Options are:
            - 'parallel flow': Fluids flow in the same direction
            - 'counter-current flow': Fluids flow in opposite directions (default)
            - 'cross flow': Fluids flow perpendicularly
            Default is 'counter-current flow' which typically provides the highest effectiveness.
        """
        self.fluid_1_input_port_name = f'{name}_fluid_1_input_port'
        self.fluid_1_output_port_name = f'{name}_fluid_1_output_port'
        self.fluid_2_input_port_name = f'{name}_fluid_2_input_port'
        self.fluid_2_output_port_name = f'{name}_fluid_2_output_port'
        self.heat_exchanger_type = heat_exchanger_type
        self.A = exchange_surface
        self.U = heat_exchange_coefficient
        super().__init__(name, ports_info = {
            self.input_port_name: 'fluid',
            self.output_port_name: 'fluid'
        })  # no external time series
        
    def balance(self, state: SimulationState, action=None):
        """
        Calculate the thermal balance of the heat exchanger for the current time step.
        
        This method implements the epsilon-NTU method to determine:
        1. The effective heat capacity rates (C) for both fluid streams
        2. The heat exchanger effectiveness (epsilon) based on NTU and flow configuration
        3. The heat transfer rate between the fluids
        4. The outlet temperatures and heat flows for both fluid streams
        
        The convention used: if inlet temperature of fluid 1 > inlet temperature of fluid 2,
        then Qdot is positive (heat flows from fluid 1 to fluid 2).
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state containing environmental and time information
        action : optional
            Unused parameter for interface compatibility
        
        Returns
        -------
        bool
            True if the simulation was successful, False if either mass flow rate is None
        list
            List of output port names that were updated during this time step
        """
        # Implements the eps NTU method
        mfr_1 = return_not_none(self.ports[self.fluid_1_input_port_name].flow['mass'], self.ports[self.fluid_1_output_port_name].flow['mass'])
        mfr_2 = return_not_none(self.ports[self.fluid_2_input_port_name].flow['mass'], self.ports[self.fluid_2_output_port_name].flow['mass'])
        # If any of the two is None, the component is not ready to be simulated
        if mfr_1 is None or mfr_2 is None:
            return False, []
        self.ports[self.fluid_1_output_port_name].flow['mass'] = mfr_1
        self.ports[self.fluid_2_output_port_name].flow['mass'] = mfr_2
        cmin = min(mfr_1, mfr_2) * WATER.cp
        cmax = max(mfr_1, mfr_2) * WATER.cp
        epsilon = self.calculate_epsilon(cmin, cmax)
        # Convention: if T1_in > T2_in, then Qdot is positive
        Qdot = epsilon * cmin * self.ports[self.fluid_1_input_port_name].T - self.ports[self.fluid_2_input_port_name].T
        self.ports[self.fluid_1_output_port_name].flow['heat'] = self.ports[self.fluid_1_input_port_name].flow['heat'] - Qdot
        self.ports[self.fluid_2_output_port_name].flow['heat'] = self.ports[self.fluid_2_input_port_name].flow['heat'] + Qdot
        self.ports[self.fluid_1_output_port_name].T = self.ports[self.fluid_1_input_port_name].T - Qdot / (mfr_1 * WATER.cp)
        self.ports[self.fluid_2_output_port_name].T = self.ports[self.fluid_2_input_port_name].T + Qdot / (mfr_2 * WATER.cp)
        return True, [self.fluid_1_output_port_name, self.fluid_2_output_port_name]

    def calculate_epsilon(self, cmin, cmax):
        """
        Calculate the heat exchanger effectiveness using the epsilon-NTU method.
        
        The effectiveness is computed based on the Number of Transfer Units (NTU) and
        the heat capacity ratio (Cr), which are then used with the appropriate formula
        depending on the flow configuration.
        
        Equations:
        - NTU = U*A / C_min (Number of Transfer Units)
        - Cr = C_min / C_max (Heat capacity ratio)
        
        Effectiveness equations by flow type:
        - Parallel flow: epsilon = (1 - exp(-NTU(1+Cr))) / (1 + Cr)
        - Counter-current: epsilon = (1 - exp(-NTU(1+Cr))) / (1 - Cr*exp(-NTU(1+Cr)))
        - Cross flow: Not yet implemented
        
        Parameters
        ----------
        cmin : float
            Minimum heat capacity rate (mass flow rate * specific heat) in W/K
        cmax : float
            Maximum heat capacity rate (mass flow rate * specific heat) in W/K
        
        Returns
        -------
        float
            Heat exchanger effectiveness (dimensionless, typically 0.0-1.0)
        """
        NTU = self.U * self.A / cmin
        cr = cmin / cmax
        match self.heat_exchanger_type:
            case 'parallel flow':
                return (1 - np.exp(-NTU * (1 + cr))) / (1 + cr)
            case 'counter-current flow':
                return (1 - np.exp(-NTU * (1 + cr))) / (1 - cr * np.exp(-NTU * (1 + cr)))


class HeatExchangerCoilTank(ImplicitComponent):
    """
    Fluid-to-fluid heat exchanger model using the epsilon-NTU method.
    Meant for a heat exchanger located in a storage tank.
    
    This component models a heat exchanger that transfers thermal energy between two fluid streams.
    The model uses the epsilon-NTU (effectiveness-Number of Transfer Units) method to calculate
    the heat transfer rate and outlet temperatures for both fluids. It supports three flow
    configurations: parallel flow, counter-current flow, and cross flow.
    
    The epsilon-NTU method is particularly useful for cases where the outlet temperatures are
    not known in advance and need to be calculated based on the effectiveness of the heat exchanger.
    
    Attributes
    ----------
    fluid_1_input_port_name : str
        Name of the input port for fluid 1
    fluid_1_output_port_name : str
        Name of the output port for fluid 1

    A : float
        Heat exchange surface area (m²)
    U : float
        Overall heat transfer coefficient (W/(m²·K))
    """

    def __init__(self, name, exchange_surface: float, heat_exchange_coefficient: float, storage_tank_name: float):
        """
        Initialize a heat exchanger component.
        
        Parameters
        ----------
        name : str
            Unique identifier for the heat exchanger
        exchange_surface : float
            Heat exchange surface area in m²
        heat_exchange_coefficient : float
            Overall heat transfer coefficient in W/(m²·K)
        temperature_sensor_name : float
            The name of the temperature sensor that provides the fixed temperature on the other (non fluid) side of the heat exchanger
        """
        self.fluid_input_port_name = f'{name}_fluid_1_input_port'
        self.fluid_output_port_name = f'{name}_fluid_1_output_port'
        self.heat_port_name = f'{name}_heat_port'
        self.A = exchange_surface
        self.U = heat_exchange_coefficient
        self.storage_tank_name = storage_tank_name
        super().__init__(name, ports_info = {
            self.input_port_name: 'fluid',
            self.output_port_name: 'fluid'
        })

    def initialize(self, ctx: InitContext):
        self.storage_tank = ctx.get_component(self.storage_tank_name)
        
    def balance(self, state: SimulationState, action=None):
        """
        Calculate the thermal balance of the heat exchanger for the current time step.
        
        This method implements the epsilon-NTU method to determine:
        1. The effective heat capacity rates (C) for both fluid streams
        2. The heat exchanger effectiveness (epsilon) based on NTU and flow configuration
        3. The heat transfer rate between the fluids
        4. The outlet temperatures and heat flows for both fluid streams
        
        The convention used: if inlet temperature of fluid 1 > inlet temperature of fluid 2,
        then Qdot is positive (heat flows from fluid 1 to fluid 2).
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state containing environmental and time information
        action : optional
            Unused parameter for interface compatibility
        
        Returns
        -------
        bool
            True if the simulation was successful, False if either mass flow rate is None
        list
            List of output port names that were updated during this time step
        """
        # Implements the eps NTU method
        mfr = return_not_none(self.ports[self.fluid_1_input_port_name].flow['mass'], self.ports[self.fluid_1_output_port_name].flow['mass'])
        # If any of the two is None, the component is not ready to be simulated
        if mfr is None:
            return False, []
        self.ports[self.fluid_output_port_name].flow['mass'] = mfr
        cmin = mfr * WATER.cp
        NTU = self.U * self.A / cmin
        epsilon = 1 - np.exp(-NTU)
        # Convention: if T1_in > T2_in, then Qdot is positive
        if isinstance(self.storage_tank, MultiNodeHotWaterTank):
            # iterative calculation of Qdot
            pass
        else:
            Qdot = epsilon * cmin * (self.ports[self.fluid_input_port_name].T - self.storage_tank.temperature)
        self.ports[self.fluid_output_port_name].flow['heat'] = self.ports[self.fluid_1_input_port_name].flow['heat'] - Qdot
        self.ports[self.fluid_output_port_name].T = self.ports[self.fluid_1_input_port_name].T - Qdot / (mfr * WATER.cp)
        self.ports[self.heat_port_name].T = self.storage_tank.temperature + log_mean_temperature_difference(
                                                                                                    self.ports[self.fluid_input_port_name].T, 
                                                                                                    self.ports[self.fluid_output_port_name].T, 
                                                                                                    self.storage_tank.temperature, 
                                                                                                    self.storage_tank.temperature)
        return True, [self.fluid_1_output_port_name, self.fluid_2_output_port_name, self.heat_port_name]



def log_mean_temperature_difference(T1_in: float, T1_out: float, T2_in: float, T2_out: float, hex_type: HeatExchangerType = 'counter-current flow'):
    """
    Calculates the log mean temperature difference between two fluids in a heat exchanger
    """
    match hex_type:
        case 'counter-current flow':
            return (T1_in - T2_out) - (T1_out - T2_in) / log((T1_in - T2_out) / (T1_out - T2_in))
        case 'parallel flow':
            return (T1_in - T2_in) - (T1_out - T2_out) / log((T1_in - T2_in) / (T1_out - T2_out))