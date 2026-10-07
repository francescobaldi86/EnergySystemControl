from energy_system_control.components.base import StorageUnit
from energy_system_control.helpers import *
from energy_system_control.core.base_classes import InitContext
from energy_system_control.constants import WATER, METHANE, CARBON_DIOXIDE
from energy_system_control.sim.state import SimulationState
from typing import Dict, List
from scipy.linalg import solve_banded
import warnings, math

class BiogasStorage(StorageUnit):

    def __init__(self,name: str,capacity: float,initial_ch4_mass: float = 0.0,initial_co2_mass: float = 0.0,):
        
        self.capacity = capacity
        self.ch4_mass = initial_ch4_mass
        self.co2_mass = initial_co2_mass
        self.SOC_0 = self.total_mass / self.biogas_density / self.capacity

        self.input_port_name = f"{name}_input"
        self.output_port_name = f"{name}_output"

        super().__init__(name,{self.input_port_name: "biogas",self.output_port_name: "biogas"})

    @property
    def total_mass(self) -> float:
        return self.ch4_mass + self.co2_mass

    @property
    def methane_mass_fraction(self) -> float:
        if self.total_mass <= 0:
            return 0.0
        return self.ch4_mass / self.total_mass

    @property
    def methane_fraction(self) -> float:
        if self.total_mass <= 0:
            return 0.0
        n_ch4 = (self.ch4_mass / METHANE.MW)
        n_co2 = self.co2_mass / CARBON_DIOXIDE.MW
        total_moles = n_ch4 + n_co2
        return n_ch4 / total_moles if total_moles > 0 else 0.0

    @property
    def biogas_density(self) -> float:
        x_ch4 = self.methane_fraction
        x_co2 = 1.0 - x_ch4
        return (x_ch4 * METHANE.rho + x_co2 * CARBON_DIOXIDE.rho)

    @property
    def biogas_LHV(self) -> float:
        """ LHV in [kJ/kg]. """
        return self.methane_mass_fraction * METHANE.LHV

    def initialize(self, ctx: InitContext):
        output_port = self.ports[self.output_port_name]
        output_port.methane_fraction = self.methane_fraction
        output_port.LHV = self.biogas_LHV
        output_port.density = self.biogas_density
        super().initialize(ctx)

    def step(self, state, action=None):
        dt = state.time_step
        input_port = self.ports[self.input_port_name]
        output_port = self.ports[self.output_port_name]

        # 1. INGRESSO
        mass_flow_in = max(0.0, input_port.flows.get("mass", 0.0))  
        methane_fraction_in = getattr(input_port, "methane_fraction", 0.0)

        # if mass_flow_in > 0:  # Perchè solo se il mass_flow_in non è zero?
        density_in = (methane_fraction_in * METHANE.rho+(1.0 - methane_fraction_in) * CARBON_DIOXIDE.rho)
        methane_mass_frac_in = (methane_fraction_in * METHANE.rho) / density_in if density_in > 0 else 0.0 #da vol a massica
        
        # Massa entrante nell'intervallo dt [kg]
        delta_mass_in = mass_flow_in * dt
        ch4_in = delta_mass_in * methane_mass_frac_in
        co2_in = delta_mass_in - ch4_in

        self.ch4_mass += ch4_in
        self.co2_mass += co2_in

        # 2. USCITA (Gestita dalla richiesta del componente a valle o da azione)
        energy_flow_out = max(0.0, output_port.flows.get("chemical_energy", 0.0))
        mass_flow_out = energy_flow_out * self.biogas_LHV
        delta_mass_out = mass_flow_out * dt

        if delta_mass_in + delta_mass_out > self.total_mass:
            delta_mass_out = -(self.total_mass + delta_mass_in)
            mass_flow_out = delta_mass_out / dt if dt > 0 else 0.0

        if delta_mass_out < 0:
            w_ch4 = self.methane_mass_fraction
            self.ch4_mass -= delta_mass_out * w_ch4
            self.co2_mass -= delta_mass_out * (1.0 - w_ch4)

    def set_inherited_port_values(self, state: SimulationState):
        # First update values of the output port
        output_port = self.ports[self.output_port_name]
        output_port.methane_fraction = self.methane_fraction
        output_port.LHV = self.biogas_LHV
        output_port.density = self.biogas_density
        return [self.output_port_name]