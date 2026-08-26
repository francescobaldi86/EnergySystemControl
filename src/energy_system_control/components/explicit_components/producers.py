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


'''class AnaerobicDigester(Producer):

    # Densità gas puri a circa:
    # 1 atm, 20-25 °C
    CH4_DENSITY = 0.72   # kg/m3
    CO2_DENSITY = 1.98   # kg/m3

    # Potere calorifico inferiore del metano puro
    # ~35.8 MJ/m3
    CH4_LHV = 35.8e6     # J/m3

    def __init__(
        self,
        name: str,
        biogas_rate: float,      # kg/s
        methane_fraction: float
    ):

        self.biogas_rate = biogas_rate
        self.methane_fraction = methane_fraction

        super().__init__(name, "biogas")

    def calculate_biogas_density(self) -> float:
        """
        Calcola la densità del biogas [kg/m3]
        assumendo miscela CH4 + CO2
        """

        return (
            self.methane_fraction * self.CH4_DENSITY
            + (1 - self.methane_fraction) * self.CO2_DENSITY
        )

    def calculate_biogas_LHV(self) -> float:
        """
        Calcola il LHV del biogas [J/m3]
        in funzione della frazione di metano
        """

        return (
            self.methane_fraction
            * self.CH4_LHV
        )

    def step(self, state, action=None):

        # PORTATA MASSICA

        # [kg/s]
        biogas_mass_flow = self.biogas_rate

        # DENSITÀ BIOGAS

        # [kg/m3]
        biogas_density = self.calculate_biogas_density()


        # PORTATA VOLUMETRICA
       # [m3/s]
        biogas_volume_flow_m3s = (
            biogas_mass_flow / biogas_density
        )

        # [L/h]
        biogas_volume_flow_lh = (
            biogas_volume_flow_m3s
            * 1000
            * 3600
        )

        # LHV BIOGAS

        # [J/m3]
        biogas_LHV = self.calculate_biogas_LHV()

        # ENERGIA CHIMICA
        # [W] = [J/s]
        chemical_energy_flow = (
            biogas_volume_flow_m3s
            * biogas_LHV
        )

        # UPDATE PORT

        port = self.ports[self.port_name]

        port.methane_fraction = self.methane_fraction
        port.LHV = biogas_LHV

        port.flows["mass"] = -biogas_mass_flow              # kg/s
#        port.flows["volume"] = -biogas_volume_flow_lh       # L/h
        port.flows["chemical_energy"] = -chemical_energy_flow  # W'''


class AnaerobicDigester(Producer):

    def __init__(
        self,
        name: str,
        biogas_rate: float,      # kg/s
        methane_fraction: float
    ):

        super().__init__(name, "biogas")

        self.biogas_rate = biogas_rate
        self.methane_fraction = methane_fraction

    @property
    def biogas_density(self):

        return (
            self.methane_fraction * Methane.rho +
            (1 - self.methane_fraction) * CarbonDioxide.rho
        )

    @property
    def biogas_LHV(self):
        """
        LHV massico del biogas [J/kg]
        """

        methane_mass_fraction = (
            self.methane_fraction * Methane.rho
        ) / self.biogas_density

        return methane_mass_fraction * Methane.LHV

    def step(self, state, action=None):

        mass_flow = self.biogas_rate                    # kg/s

        volume_flow = (
            mass_flow / self.biogas_density
        )

        chemical_energy = (
            mass_flow * self.biogas_LHV
        )

        port = self.ports[self.port_name]

        port.methane_fraction = self.methane_fraction
        port.LHV = self.biogas_LHV

        port.flows["mass"] = -mass_flow
        port.flows["volume"] = -volume_flow
        port.flows["chemical_energy"] = -chemical_energy
