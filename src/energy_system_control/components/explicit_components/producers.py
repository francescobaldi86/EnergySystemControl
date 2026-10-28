import pandas as pd
import os

from energy_system_control.components.base import ExplicitComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.constants.fluids import Methane, CarbonDioxide
from energy_system_control.components.base import TimeSeriesData



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
    def volume_flow_l_h(self) -> float:
     """Biogas volume flow [L/h]."""
     return self.biogas_rate

    @property
    def volume_flow_m3_s(self) -> float:
      """Biogas volume flow [m³/s]."""
      return self.volume_flow_l_h / 3_600_000.0

    @property
    def mass_flow(self) -> float:
     """Biogas mass flow [kg/s]."""
     return self.volume_flow_m3_s * self.biogas_density

    @property
    def chemical_power(self) -> float:
        """kg/s * kJ/kg = kJ/s = kW"""

        return self.mass_flow * self.biogas_LHV

    def step(self, state, action=None):

        mass_flow = self.mass_flow

        volume_flow = self.volume_flow_m3_s

        chemical_power = self.chemical_power

        port = self.ports[self.port_name]

        port.methane_fraction = self.methane_fraction
        port.LHV = self.biogas_LHV

        port.flows["mass"] = -mass_flow
        port.flows["volume"] = -volume_flow
        port.flows["chemical_energy"] = -chemical_power


class AnaerobicDigesterFromData(AnaerobicDigester):
    """
    Biogas storage with time-dependent biogas composition
    and production data read from a CSV file.
    """

    def __init__(self,
                 name: str, 
                 data_path: str, 
                 filename: str, 
                 date_format: str = "%Y%m%d:%H%M", 
                 rescale_factor: float = 1.0, 
                 mass_flow_column_name: str = 'mass_flow',
                 methane_fraction_column_name: str = 'methane_fraction',
                 mass_flow_unit: str = 'kg/s',
                 methane_fraction_unit: str = 'adimensional'):
        # Inizializza il componente di accumulo

        # Legge il file CSV
        data = pd.read_csv(os.path.join(data_path, filename),sep=";",decimal=".",index_col=0,)

        # Converte l'indice in date
        data["time"] = pd.to_datetime(data.index,format=date_format,utc=True,)

        data = data.set_index("time").sort_index()
        data_mass_flow = data[mass_flow_column_name]
        data_methane_fraction = data[methane_fraction_column_name]
        data_mass_flow *= rescale_factor
        ts_mass_flow = TimeSeriesData(
                                raw = data_mass_flow,
                                var_type = 'mass_flow',
                                var_unit = mass_flow_unit,
                            )
        ts_methane_fraction = TimeSeriesData(
                                raw = data_methane_fraction,
                                var_type = 'adimensional',
                                var_unit = methane_fraction_unit,
                            )

        # Conserva le serie temporali
        self.methane_fraction_data = ts_methane_fraction
        self.mass_flow_data = ts_mass_flow

        super().__init__(name=name,biogas_rate = 0.0, methane_fraction = 0.0)

    def resample_data(self, time_step_h: float, sim_end_h: float):
        self.methane_fraction_data.resample(time_step_h=time_step_h, sim_end_h=sim_end_h)
        self.mass_flow_data.resample(time_step_h=time_step_h, sim_end_h=sim_end_h)

    def step(self, state: SimulationState, action = None):

        # recupero dai dati TimeSeries data la frazione di metano e la portata in massa di produzione di biogas
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

        port.flows["mass"] = -mass_flow
        port.flows["volume"] = -volume_flow_m3_s
        port.flows["chemical_energy"] = -chemical_power