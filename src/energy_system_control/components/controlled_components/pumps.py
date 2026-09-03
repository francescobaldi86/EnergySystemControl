from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import *
from energy_system_control.constants import WATER
from abc import abstractmethod


class SimplePump(ControlledComponent):
    """
    A simple pump that provides constant mass flow rate control.
    
    This pump maintains a fixed maximum mass flow rate and modulates it based on a control
    action signal (0 to 1). The pump transfers fluid from input to output port without
    temperature change, suitable for systems where pressure rise and electrical consumption
    are negligible or unmodeled.
    
    Attributes
    ----------
    name : str
        Unique identifier for the pump
    mass_flow_rate : float
        Maximum mass flow rate in kg/s
    input_water_port_name : str
        Name of the fluid input port
    output_water_port_name : str
        Name of the fluid output port
    """
    
    def __init__(self, name: str, mass_flow_rate: float):
        """
        Initialize a simple pump with constant mass flow rate.
        
        Parameters
        ----------
        name : str
            Unique identifier for the pump
        mass_flow_rate : float
            Maximum mass flow rate in kg/s that the pump can deliver
        """
        self.name = name
        self.mass_flow_rate = mass_flow_rate
        self.input_water_port_name = f'{name}_water_input_port'
        self.output_water_port_name = f'{name}_water_output_port'
        ports_info = {
            self.input_water_port_name: 'fluid',
            self.output_water_port_name: 'fluid',
        }
        super().__init__(name, ports_info)

    def step(self, state: SimulationState, action=None):
        """
        Execute the pump for one simulation time step.
        
        Sets the mass flow rate based on the control action (0-1), and passes the fluid
        through from input to output at the same temperature. Heat flow is calculated
        based on mass flow rate and fluid temperature.
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state
        action : float, optional
            Control signal (0-1) that modulates the pump flow rate. 
            Actual flow = mass_flow_rate * action. Default is None.
        """
        flow = self.mass_flow_rate * action
        self.ports[self.input_water_port_name].flows['mass'] = flow
        self.ports[self.input_water_port_name].flows['heat'] = flow * WATER.cp * self.ports[self.input_water_port_name].T
        self.ports[self.output_water_port_name].flows['mass'] = flow
        self.ports[self.output_water_port_name].flows['heat'] = self.ports[self.input_water_port_name].flows['heat']
        self.ports[self.output_water_port_name].T = self.ports[self.input_water_port_name].T



class Pump(ControlledComponent):
    """
    Abstract base class for variable flow rate pumps with optional electricity consumption.
    
    This class provides the framework for pumps that can modulate their flow rate based on
    system conditions. Subclasses must implement methods to determine flow rate, pressure
    rise, and pump efficiency. The pump can optionally include electrical power input to
    model the pump's electrical consumption.
    
    The pump operates by:
    1. Calculating the desired flow rate via get_flow() method
    2. If electricity tracking is enabled:
       - Calculating required pressure rise via get_pressure_rise()
       - Calculating pump efficiency via get_efficiency()
       - Computing electrical power: P_elec = (flow * pressure_rise) / efficiency
    3. Transferring fluid from input to output at constant temperature
    
    Attributes
    ----------
    name : str
        Unique identifier for the pump
    input_water_port_name : str
        Name of the fluid input port
    output_water_port_name : str
        Name of the fluid output port
    electricity_port_name : str or None
        Name of the electricity input port (None if electricity tracking is disabled)
    """

    def __init__(self, name: str, include_electricity_input: bool=False):
        """
        Initialize a pump component.
        
        Parameters
        ----------
        name : str
            Unique identifier for the pump
        include_electricity_input : bool, optional
            If True, creates an electricity input port and enables electrical consumption
            calculations. Default is False.
        """
        self.name = name
        self.input_water_port_name = f'{name}_water_input_port'
        self.output_water_port_name = f'{name}_water_output_port'
        ports_info = {
            self.input_water_port_name: 'fluid',
            self.output_water_port_name: 'fluid',
        }
        self.electricity_port_name = None
        if include_electricity_input:
            self.electricity_port_name = f'{name}_electricity_port'
            ports_info[self.electricity_port_name] = 'electricity'
        super().__init__(name, ports_info)



    def step(self, state: SimulationState, action=None):
        """
        Execute the pump for one simulation time step.
        
        This method:
        1. Gets the current flow rate from the subclass implementation
        2. If electricity tracking is enabled:
           - Calculates pressure rise and efficiency
           - Computes electrical power consumption
        3. Updates input and output ports with flow rate and temperature
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state
        action : optional
            Control action passed to subclass methods (specific meaning depends on
            the subclass implementation)
        """
        flow = self.get_flow(state, action)
        if self.electricity_port_name is not None:
            pressure_rise = self.get_pressure_rise(state, action)
            efficiency = self.get_efficiency(state, action)
            self.ports[self.electricity_port_name].flows['electricity'] = flow * pressure_rise / efficiency
        self.ports[self.input_water_port_name].flows['mass'] = flow
        self.ports[self.input_water_port_name].flows['heat'] = flow * WATER.cp * self.ports[self.input_water_port_name].T
        self.ports[self.output_water_port_name].flows['mass'] = flow
        self.ports[self.output_water_port_name].flows['heat'] = self.ports[self.input_water_port_name].flows['heat']
        self.ports[self.output_water_port_name].T = self.ports[self.input_water_port_name].T
        
    @abstractmethod
    def get_pressure_rise(self, state: SimulationState, action):
        """
        Calculate the pressure rise provided by the pump.
        
        This method must be implemented by subclasses to return the pressure rise (Pa)
        that the pump provides at the current operating conditions.
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state
        action : optional
            Control action affecting pressure rise calculation
        
        Returns
        -------
        float
            Pressure rise in Pascals (Pa)
        """
        raise NotImplementedError

    @abstractmethod
    def get_efficiency(self, state: SimulationState, action):
        """
        Calculate the pump efficiency.
        
        This method must be implemented by subclasses to return the overall efficiency
        of the pump (0-1) at the current operating conditions.
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state
        action : optional
            Control action affecting efficiency calculation
        
        Returns
        -------
        float
            Pump efficiency (dimensionless, typically 0.0-1.0)
        """
        raise NotImplementedError

    @abstractmethod
    def get_flow(self, state: SimulationState, action):
        """
        Calculate the current mass flow rate of the pump.
        
        This method must be implemented by subclasses to return the mass flow rate (kg/s)
        based on the current system state and control action.
        
        Parameters
        ----------
        state : SimulationState
            Current simulation state
        action : optional
            Control action affecting flow rate calculation
        
        Returns
        -------
        float
            Mass flow rate in kg/s
        """
        raise NotImplementedError