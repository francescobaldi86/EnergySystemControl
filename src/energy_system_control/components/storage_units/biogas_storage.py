from energy_system_control.components.base import StorageUnit
from energy_system_control.helpers import *
from energy_system_control.core.base_classes import InitContext
from energy_system_control.constants import METHANE, CARBON_DIOXIDE
from energy_system_control.sim.state import SimulationState

class BiogasStorage(StorageUnit):
    """
    Biogas storage with CH4 and CO2 mass balances, 
    equipped with an output port for the engine and one for the emergency flare.
    """

    def __init__(self, name: str, capacity: float, initial_soc: float = 0.0, initial_methane_fraction: float = 0.5):
        
        # 1. INPUT VALIDATION
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if not 0.0 <= initial_soc <= 1.0:
            raise ValueError("initial_soc must be between 0 and 1.")
        if not 0.0 <= initial_methane_fraction <= 1.0:
            raise ValueError("initial_methane_fraction must be between 0 and 1.")

        self.capacity = float(capacity)
        self.SOC_0 = float(initial_soc)
        self.SOC = float(initial_soc)

        # 2. DEFINIZIONE DELLE PORTE
        self.input_port_name = f"{name}_input"
        self.output_port_name = f"{name}_output"
        self.flare_port_name = f"{name}_flare_output"

        # REGISTRAZIONE DELLE 3 PORTE NELLA CLASSE MADRE
        super().__init__(name, {
            self.input_port_name: "biogas", 
            self.output_port_name: "biogas",
            self.flare_port_name: "biogas"
        })

        # 3. INITIAL GAS COMPOSITION
        x_ch4 = float(initial_methane_fraction)
        x_co2 = 1.0 - x_ch4

        self._initial_density = (x_ch4 * METHANE.rho + x_co2 * CARBON_DIOXIDE.rho)

        # 4. INITIAL MASS CALCULATION
        initial_volume = self.SOC_0 * self.capacity
        initial_total_mass = (initial_volume * self._initial_density)

        mean_molecular_weight = (x_ch4 * METHANE.MW + x_co2 * CARBON_DIOXIDE.MW)
        if mean_molecular_weight > 0:
            initial_ch4_mass_fraction = (x_ch4 * METHANE.MW / mean_molecular_weight)
        else:
            initial_ch4_mass_fraction = 0.0

        self.ch4_mass = (initial_total_mass * initial_ch4_mass_fraction)
        self.co2_mass = (initial_total_mass - self.ch4_mass)


    # --- PROPERTIES ---

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
        n_ch4 = self.ch4_mass / METHANE.MW
        n_co2 = self.co2_mass / CARBON_DIOXIDE.MW
        total_moles = n_ch4 + n_co2
        if total_moles <= 0:
            return 0.0
        return n_ch4 / total_moles

    @property
    def biogas_density(self) -> float:
        if self.total_mass <= 0:
            return (0.5 * METHANE.rho + 0.5 * CARBON_DIOXIDE.rho)
        x_ch4 = self.methane_fraction
        x_co2 = 1.0 - x_ch4
        return (x_ch4 * METHANE.rho + x_co2 * CARBON_DIOXIDE.rho)

    @property
    def biogas_LHV(self) -> float:
        return (self.methane_mass_fraction * METHANE.LHV)

    @property
    def volume_m3(self) -> float:
        density = self.biogas_density
        if density <= 0:
            return 0.0
        return self.total_mass / density


    # --- PORT UPDATE & INITIALIZATION ---

    def _update_output_port(self):
        """Propaga le proprietà del gas su ENTRAMBE le porte di uscita."""
        for p_name in [self.output_port_name, self.flare_port_name]:
            if p_name in self.ports:
                port = self.ports[p_name]
                port.methane_fraction = self.methane_fraction
                port.LHV = self.biogas_LHV
                port.density = self.biogas_density

    def initialize(self, ctx: InitContext):
        self._update_soc()
        self._update_output_port()
        super().initialize(ctx)
        self._update_output_port()

    def set_inherited_port_values(self, state: SimulationState):
        self._update_output_port()
        output = {}
        # Segnala all'ambiente le proprietà per entrambe le porte (Motore e Torcia)
        for p_name in [self.output_port_name, self.flare_port_name]:
            if p_name in self.ports:
                output[p_name] = {
                    'methane_fraction': self.methane_fraction,
                    'LHV': self.biogas_LHV,
                    'density': self.biogas_density
                }
        return output

    def _update_soc(self):
        if self.capacity <= 0:
            self.SOC = 0.0
            return
        self.SOC = min(1.0, max(0.0, self.volume_m3 / self.capacity))


    # --- SIMULATION STEP ---

    def step(self, state, action=None):
        dt = state.time_step
        if dt < 0:
            raise ValueError("time_step cannot be negative.")

        input_port = self.ports.get(self.input_port_name)
        output_port = self.ports.get(self.output_port_name)
        flare_port = self.ports.get(self.flare_port_name)

        w_ch4_current = self.methane_mass_fraction
        lhv_current = self.biogas_LHV

        # A. GAS INLET (Dal digestore)
        mass_flow_in = max(0.0, input_port.flows.get("mass", 0.0) or 0.0)
        x_ch4_in = getattr(input_port, "methane_fraction", 0.0)
        x_ch4_in = min(1.0, max(0.0, x_ch4_in))
        x_co2_in = 1.0 - x_ch4_in

        mean_mw_in = (x_ch4_in * METHANE.MW + x_co2_in * CARBON_DIOXIDE.MW)
        if mean_mw_in > 0:
            w_ch4_in = (x_ch4_in * METHANE.MW / mean_mw_in)
        else:
            w_ch4_in = 0.0

        delta_mass_in = mass_flow_in * dt

        # B. GAS OUTLET (Somma prelievi: Motore + Torcia)
        chem_energy_engine = (output_port.flows.get("chemical_energy", 0.0) or 0.0) if output_port else 0.0
        chem_energy_flare = (flare_port.flows.get("chemical_energy", 0.0) or 0.0) if flare_port else 0.0
        
        total_energy_flow_out = max(0.0, -chem_energy_engine) + max(0.0, -chem_energy_flare)
        
        if lhv_current > 0:
            mass_flow_out = total_energy_flow_out / lhv_current
        else:
            mass_flow_out = 0.0

        delta_mass_out = min(mass_flow_out * dt, self.total_mass)

        # C. MASS BALANCES UPDATE
        self.ch4_mass += (delta_mass_in * w_ch4_in) - (delta_mass_out * w_ch4_current)
        self.co2_mass += (delta_mass_in * (1.0 - w_ch4_in)) - (delta_mass_out * (1.0 - w_ch4_current))
        
        self._update_soc()