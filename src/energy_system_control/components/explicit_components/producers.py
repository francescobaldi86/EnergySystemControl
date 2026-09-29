from energy_system_control.components.base import ExplicitComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.constants.fluids import Methane, CarbonDioxide



class Producer(ExplicitComponent):
    port_name: str
    production_type: str
    def __init__(self, name: str, production_type: str):
        self.production_type = production_type
        self.port_name = f'{name}_{self.production_type}_port'
        super().__init__(name, {self.port_name: self.production_type})


class ConstantPowerProducer(Producer):
    def __init__(self, name: str, production_type: str, power: float):
        super().__init__(name, production_type)
        self.power = power  # 
    
    def step(self, state: SimulationState, action = None): 
        self.ports[self.port_name].flows[self.production_type] = -self.power  # Since it is a producer, the net energy flow is always negative

class AnaerobicDigester(Producer):

    def __init__(self,name: str,biogas_rate: float,methane_fraction: float):
        """
        Model of an anerobic digster with constant biogas production

        Parameters
        ----------
        name : str
            Name of the component
        biogas_rate : float
            Constant production rate of the digester, in [m3/h]
        methane_fraction : float
            Constant methane fraction of the biogas produced, in [vol/vol]. Must be between 0 and 1
        """
        super().__init__(name, "biogas")

        if not 0.0 <= methane_fraction <= 1.0:
            raise ValueError("methane_fraction must be between 0 and 1")

        if biogas_rate < 0:
            raise ValueError("biogas_rate must be >= 0")

        self.biogas_rate = biogas_rate
        self.methane_fraction = methane_fraction

    @property
    def biogas_density(self) -> float:

        x_ch4 = self.methane_fraction
        x_co2 = 1.0 - x_ch4

        return (x_ch4 * Methane.rho + x_co2 * CarbonDioxide.rho)

    @property
    def methane_mass_fraction(self) -> float:
       
        return (self.methane_fraction * Methane.rho) / self.biogas_density

    @property
    def biogas_LHV(self) -> float:
        """[kJ/kg] """

        return (self.methane_mass_fraction* Methane.LHV)

    @property
    def mass_flow(self) -> float:
        """Biogas mass flow [kg/s]."""

        # Nm³/h -> m³/s
        volume_flow_m3_s = self.biogas_rate / 3600.0

        return (volume_flow_m3_s * self.biogas_density)

    @property
    def chemical_power(self) -> float:
        """kg/s * kJ/kg = kJ/s = kW"""

        return self.mass_flow * self.biogas_LHV

    def step(self, state, action=None):

        mass_flow = self.mass_flow

        volume_flow = self.biogas_rate / 3600.0

        chemical_power = self.chemical_power

        port = self.ports[self.port_name]

        port.methane_fraction = self.methane_fraction
        port.LHV = self.biogas_LHV

        port.flows["mass"] = -mass_flow
        port.flows["volume"] = -volume_flow
        port.flows["chemical_energy"] = -chemical_power