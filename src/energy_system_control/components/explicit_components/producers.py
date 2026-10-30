import pandas as pd
import os

from energy_system_control.components.base import ExplicitComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.constants.fluids import Methane, CarbonDioxide
from energy_system_control.core.base_classes import TimeSeriesData
from energy_system_control.helpers import TimeAlignment


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
        # NON tocchiamo self.ports qui

    def create_ports(self):
        # 1. Chiama il metodo originale per creare materialmente la porta
        ports = super().create_ports()   
        return ports

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
        return (self.methane_mass_fraction * Methane.LHV)

    @property
    def volume_flow_l_h(self) -> float:
     """Biogas volume flow [L/h]."""
     return self.biogas_rate

    @property
    def volume_flow_m3_s(self) -> float:
      """Biogas volume flow [m³/s]."""
      return self.volume_flow_l_h / 3_600_000.0

    @property
    def mass_flow(self) -> float:
         volume_flow_m3_s = self.biogas_rate / 3600.0
         return (volume_flow_m3_s * self.biogas_density)

    @property
    def chemical_power(self) -> float:
        return self.mass_flow * self.biogas_LHV

    def step(self, state, action=None):
        '''
        for port in self.ports.values():
            for layer in port.flows.keys():
                port.flows[layer] = 0.0
        '''
        mass_flow = self.mass_flow
        volume_flow = self.volume_flow_m3_s
        chemical_power = self.chemical_power

        port = self.ports[self.port_name]
        port.methane_fraction = self.methane_fraction
        port.LHV = self.biogas_LHV

        port.flows["mass"] = -mass_flow
        port.flows["volume"] = -volume_flow
        port.flows["chemical_energy"] = -chemical_power

    def set_inherited_port_values(self, state: SimulationState):
        """
        Propaga le caratteristiche termodinamiche del biogas generato (frazione di metano, 
        potere calorifico inferiore e densità) alla porta connessa a valle (es. BiogasStorage).
        """
        port = self.ports[self.port_name]
        
        # Aggiornamento locale (opzionale ma consigliato per consistenza)
        port.methane_fraction = self.methane_fraction
        port.LHV = self.biogas_LHV
        port.density = self.biogas_density
        
        # Restituzione nel nuovo formato dizionario annidato
        return {
            self.port_name: {
                'methane_fraction': self.methane_fraction,
                'LHV': self.biogas_LHV,
                'density': self.biogas_density
            }
        }

class AnaerobicDigesterFromData(AnaerobicDigester):
    """
    Biogas storage with time-dependent biogas composition
    and production data read from a CSV file.
    """

    def __init__(self,
                 name: str, 
                 data_path: str, 
                 filename: str,
                 time_alignment: TimeAlignment = "yearly",
                 date_format: str = "%Y%m%d:%H%M", 
                 rescale_factor: float = 1.0, 
                 volume_flow_column_name: str = "Produzione biogas[L/h]",
                 methane_fraction_column_name: str = "contenuto metano in biogas prodotto",
            ):
        # Inizializza il componente di accumulo

        # Legge il file CSV
        data = pd.read_csv(os.path.join(data_path, filename),sep=";",decimal=".",index_col=0,)

        # Converte l'indice in date
        data["time"] = pd.to_datetime(data.index,format=date_format,utc=True,)

        data = data.set_index("time").sort_index()
        data_volume_flow = data[volume_flow_column_name]*rescale_factor
        data_methane_fraction = data[methane_fraction_column_name]

        # Serie temporale della portata volumetrica in L/h
        self.volume_flow_data = TimeSeriesData(
            raw=data_volume_flow,
            var_type="volume_flow",
            var_unit="l/h",
            time_alignment=time_alignment,
        )

        # Serie temporale della frazione molare di metano
        self.methane_fraction_data = TimeSeriesData(
            raw=data_methane_fraction,
            var_type="adimensional",
            var_unit="adimensional",
            time_alignment=time_alignment,
        )

        super().__init__(name=name,biogas_rate = 0.0, methane_fraction = 0.0)

    def resample_data(self, time_step_h: float, simulation_end_h: float, simulation_start_datetime: str):
        self.methane_fraction_data.resample(time_step_h=time_step_h, simulation_end_h=simulation_end_h, simulation_start_datetime = simulation_start_datetime)
        self.volume_flow_data.resample(time_step_h=time_step_h, simulation_end_h=simulation_end_h, simulation_start_datetime = simulation_start_datetime)

    def step(self, state: SimulationState, action = None):

        methane_fraction = self.methane_fraction_data.data[state.time_id]
        volume_flow_l_h = self.volume_flow_data.data[state.time_id]
        # Aggiorna la frazione di metano
        self.methane_fraction = methane_fraction

        # Portata volumetrica: L/h -> m³/s
        volume_flow_m3_s = volume_flow_l_h / 3_600_000.0

        # Calcola la portata massica [kg/s]
        mass_flow = volume_flow_m3_s * self.biogas_density

        # Calcola la potenza chimica [kW]
        chemical_power = mass_flow * self.biogas_LHV

       # Assegna i valori alla porta di output
        port = self.ports[self.port_name]
        port.methane_fraction = methane_fraction
        port.LHV = self.biogas_LHV
        port.density = self.biogas_density

        port.flows["mass"] = -mass_flow
        port.flows["volume"] = -volume_flow_m3_s
        port.flows["chemical_energy"] = -chemical_power
    