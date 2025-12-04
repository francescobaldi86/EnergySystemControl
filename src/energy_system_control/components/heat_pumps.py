from energy_system_control.components.utilities import HeatSource
from energy_system_control.helpers import OnOffComponentError, C2K
from energy_system_control.sim.state import SimulationState

class HeatPump(HeatSource):
    def __init__(self, name: str, Qdot_max: float):
        super().__init__(name = name, source_type='electricity')
        self.Qdot_out = Qdot_max

    @property
    def COP(self): 
        return self.get_efficiency()
    
    def get_heat_output(self, state: SimulationState):
        return self.Qdot_out
    
    def step(self, state: SimulationState, action):
        if action not in {0.0, 1.0}:
            raise OnOffComponentError(f'The control input to the component {self.name} of type "HeatPumpConstantEfficiency" should be either 1 or 0. {action} was provided at time step {state.time}')
        return super().step(state, action)


class HeatPumpConstantEfficiency(HeatPump):
    COP_design: float
    def __init__(self, name: str, Qdot_max: float, COP = 3):
        super().__init__(name = name, Qdot_max = Qdot_max)
        self.COP_design = COP

    def get_efficiency(self, state:SimulationState):
        return self.COP_design
    
    
class HeatPumpLorentzEfficiency(HeatPump):
    COP_design: float
    eta_lorentz: float
    Wdot_design: float
    dT_air: float
    dT_water: float
    T_air_design: float
    T_water_design: float
    heat_capacity_loss: float
    def __init__(self, name, Qdot_max: float | None = None, Wdot_design: float | None = None, COP_design: float | None = None, eta_lorentz: float | None = None, T_air_design: float = 7, T_water_design: float = 40, dT_air : float = 5.0, dT_water: float = 5.0, heat_capacity_loss: float = 0.0):
        """
        Model of heat pump based on a constant second-law efficiency. Example reference is Walden, Jasper VM, and Roger Padullés. "An analytical solution to optimal heat pump integration." Energy Conversion and Management 320 (2024): 118983.
        Note that the assumption is that of a constant heat output, while the power input varies with the COP

        Parameters
        ----------
        name : str
            Name of the component
        Qdot_max : float, optional
            Design heat output [kW] of the heat pump
        Wdot_design : float, optional
            The electric power input [kW] in design conditions
        COP_design : float, optional
            The value of the COP of the heat pump [-] in design conditions
        eta_lorentz: float, optional
            The constant value of the Lorentz efficiency [-] of the heat pump
        T_air_design: float, optional
            The value of the ambient air temperature [°C] at which design conditions are calculated. Defaults to 7°C according to EU regulation 814/2013
        T_water_design: float, optional
            The value of the storage temperature [°C] at which design conditions are calculated. Defaults to 40°C 
        dT_air: float, optional
            Temperature difference [K] between the temperature of the ambient air and the evaporation temperature. Defaults to 5.0
        dt_water: float, optional
            Temperature difference [K] between the temperature of the water in the storage and the condensation temperature. Defaults to 5.0
        heat_capacity_loss: float, optional
            Fraction of the heating output lost for every additional K to the [T_condensation - T_evaporation] difference. Defaults to 0.0 (constant thermal power output)
        """
        super().__init__(name = name, Qdot_max = Qdot_max)
        self.dT_air = dT_air
        self.dT_water = dT_water
        self.heat_capacity_loss = heat_capacity_loss
        self.T_air_design = C2K(T_air_design)
        self.T_water_design = C2K(T_water_design)
        provided = (Wdot_design is not None) + (COP_design is not None) + (eta_lorentz is not None)
        if provided == 1:
            pass
        elif provided == 0:
            raise ValueError('Exactly one between Wdot_design, COP_design and eta_lorentz must be provided, while you specified none of the three')
        else:
            raise ValueError('Exactly one between Wdot_design, COP_design and eta_lorentz must be provided, while you specified more than one')
        if Wdot_design:
            self.Wdot_design = Wdot_design
            self.COP_design = self.Qdot_out / self.Wdot_design
            self.eta_lorentz = self.COP_design * self.calculate_Carnot_COP(self.T_air_design, self.T_water_design)
        elif COP_design:
            self.COP_design = COP_design
            self.Wdot_design = self.Qdot_out / self.COP_design
            self.eta_lorentz = self.COP_design / self.calculate_Carnot_COP(self.T_air_design, self.T_water_design)
        elif eta_lorentz:
            self.eta_lorentz = eta_lorentz
            self.COP_design = self.eta_lorentz * self.calculate_Carnot_COP(self.T_air_design, self.T_water_design)
            self.Wdot_design = self.Qdot_out / self.COP_design

    def calculate_Carnot_COP(self, T_air, T_water):
        return (T_water + self.dT_water) / (T_water - T_air + self.dT_air + self.dT_water)
    
    def get_heat_output(self, state: SimulationState):
        return self._get_heat_output(state.environmental_data['Temperature ambient'], self.ports[self.heat_output_port_name].T)

    def _get_heat_output(self, T_air, T_water):
        if self.heat_capacity_loss != 0.0:
            return self.Qdot_out * (1 - self.heat_capacity_loss * ((T_water - T_air) - (self.T_water_design - self.T_air_design)))
        else:
            return self.Qdot_out
        
    def get_efficiency(self, state: SimulationState):
        return self._get_efficiency(state.environmental_data['Temperature ambient'], self.ports[self.heat_output_port_name].T)
    
    def _get_efficiency(self, T_air, T_water):
        return self.eta_lorentz * self.calculate_Carnot_COP(T_air, T_water)