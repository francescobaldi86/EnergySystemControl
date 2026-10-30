from energy_system_control.components.base import StorageUnit
from energy_system_control.helpers import *
from energy_system_control.core.base_classes import InitContext
from energy_system_control.constants import METHANE, CARBON_DIOXIDE
from energy_system_control.sim.state import SimulationState

class BiogasStorage(StorageUnit):
    """
    Biogas storage with CH4 and CO2 mass balances.

    Parameters
    ----------
    name : str
        Storage unit name.
    capacity : float
        Maximum storage volume [m3].
    initial_soc : float
        Initial state of charge, between 0 and 1.
    initial_methane_fraction : float
        Initial molar fraction of methane, between 0 and 1.

    Notes
    -----
    - Port mass flows are in kg/s.
    - Port chemical energy flows are in kW, equivalent to kJ/s.
    - time_step is in seconds.
    - SOC is the occupied volume divided by the nominal capacity.
    """

    def __init__(self,name: str,capacity: float,initial_soc: float = 0.0,initial_methane_fraction: float = 0.5,):
        super().__init__(name,{f"{name}_input": "biogas",f"{name}_output": "biogas",},)

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
        self.input_port_name = f"{name}_input"
        self.output_port_name = f"{name}_output"

        # 2. INITIAL GAS COMPOSITION
        # Methane fraction is MOLAR, not mass-based.
        x_ch4 = float(initial_methane_fraction)
        x_co2 = 1.0 - x_ch4

        # Initial mixture density [kg/m3].
        self._initial_density = ( x_ch4 * METHANE.rho + x_co2 * CARBON_DIOXIDE.rho)

        # 3. INITIAL MASS CALCULATION
        initial_volume = self.SOC_0 * self.capacity
        initial_total_mass = ( initial_volume * self._initial_density)

        # Convert molar fraction into mass fraction.
        mean_molecular_weight = ( x_ch4 * METHANE.MW + x_co2 * CARBON_DIOXIDE.MW)

        if mean_molecular_weight > 0:
            initial_ch4_mass_fraction = ( x_ch4 * METHANE.MW / mean_molecular_weight)
        else:
            initial_ch4_mass_fraction = 0.0

        self.ch4_mass = (initial_total_mass * initial_ch4_mass_fraction)
        self.co2_mass = (initial_total_mass - self.ch4_mass)

    # 4. MASS AND COMPOSITION PROPERTIES

    @property
    def total_mass(self) -> float:
        """Total gas mass [kg]."""
        return self.ch4_mass + self.co2_mass

    @property
    def methane_mass_fraction(self) -> float:
        """Methane mass fraction [-]."""
        if self.total_mass <= 0:
            return 0.0
        return self.ch4_mass / self.total_mass

    @property
    def methane_fraction(self) -> float:
        """Methane molar fraction [-]."""
        n_ch4 = self.ch4_mass / METHANE.MW
        n_co2 = self.co2_mass / CARBON_DIOXIDE.MW
        total_moles = n_ch4 + n_co2
        if total_moles <= 0:
            return 0.0
        return n_ch4 / total_moles

    @property
    def biogas_density(self) -> float:
        """Biogas mixture density [kg/m3]."""
        if self.total_mass <= 0:
            return (0.5 * METHANE.rho + 0.5 * CARBON_DIOXIDE.rho)

        x_ch4 = self.methane_fraction
        x_co2 = 1.0 - x_ch4

        return (x_ch4 * METHANE.rho + x_co2 * CARBON_DIOXIDE.rho)

    @property
    def biogas_LHV(self) -> float:
        """Biogas lower heating value [kJ/kg]."""
        return (self.methane_mass_fraction * METHANE.LHV)

    @property
    def volume_m3(self) -> float:
        """Occupied gas volume [m3]."""
        density = self.biogas_density
        if density <= 0:
            return 0.0
        return self.total_mass / density

    @property
    def volume_l(self) -> float:
        """Occupied gas volume [L]."""
        return self.volume_m3 * 1000.0

    # 5. PORT INITIALIZATION

    def _update_output_port(self):
        """Update output gas properties."""
        output_port = self.ports[self.output_port_name]

        output_port.methane_fraction = self.methane_fraction
        output_port.LHV = self.biogas_LHV
        output_port.density = self.biogas_density

    def initialize(self, ctx: InitContext):
        self._update_soc()
        self._update_output_port()

        super().initialize(ctx)

        # Update again in case initialize() changes port values.
        self._update_output_port()

    def set_inherited_port_values(self,state: SimulationState,):
        self._update_output_port()
        return [self.output_port_name]

    # SOC UPDATE

    def _update_soc(self):
        """Recalculate SOC from the current occupied volume."""
        if self.capacity <= 0:
            self.SOC = 0.0
            return

        self.SOC = min(
            1.0,
            max(0.0, self.volume_m3 / self.capacity),
        )

    # SIMULATION STEP

    def step(self, state, action=None):
        dt = state.time_step

        if dt < 0:
            raise ValueError("time_step cannot be negative.")

        input_port = self.ports[self.input_port_name]
        output_port = self.ports[self.output_port_name]

        # A. GAS INLET
        mass_flow_in = max(0.0,input_port.flows.get("mass", 0.0) or 0.0,)

        # Input methane fraction is assumed to be MOLAR.
        x_ch4_in = getattr(input_port,"methane_fraction",0.0,)

        x_ch4_in = min(1.0, max(0.0, x_ch4_in))
        x_co2_in = 1.0 - x_ch4_in

        # Mixture density at the inlet [kg/m3].
        density_in = ( x_ch4_in * METHANE.rho + x_co2_in * CARBON_DIOXIDE.rho)

        # Convert molar fraction to mass fraction.
        mean_mw_in = (x_ch4_in * METHANE.MW + x_co2_in * CARBON_DIOXIDE.MW)

        if mean_mw_in > 0:
            w_ch4_in = (x_ch4_in * METHANE.MW/ mean_mw_in)
        else:
            w_ch4_in = 0.0

        # Incoming mass during the timestep [kg].
        requested_mass_in = mass_flow_in * dt

        # Available volume before accepting incoming gas.
        current_volume = self.volume_m3
        available_volume = max(0.0,self.capacity - current_volume,)

        # Approximate mass that fits in the available volume.
        max_mass_in = available_volume * density_in
        accepted_mass_in = min(requested_mass_in,max_mass_in,)
        self.ch4_mass += accepted_mass_in * w_ch4_in
        self.co2_mass += (accepted_mass_in * (1.0 - w_ch4_in))

        # Overflow is the incoming mass that was not accepted.
        self.overflow_mass = (requested_mass_in - accepted_mass_in)
        # GAS OUTLET
        # Negative chemical energy flow means energy is leaving the storage, as in the original convention.
        chemical_energy_out = (output_port.flows.get("chemical_energy", 0.0)or 0.0)

        energy_flow_out = max(0.0,-chemical_energy_out,)

        # Update mixture LHV after accepting inlet gas.
        current_lhv = self.biogas_LHV

        # Requested mass flow [kg/s].
        if current_lhv > 0:
            mass_flow_out = energy_flow_out / current_lhv
        else:
            mass_flow_out = 0.0

        requested_mass_out = mass_flow_out * dt

        # Cannot withdraw more gas than is stored.
        accepted_mass_out = min(requested_mass_out,self.total_mass,)

        if self.total_mass > 0 and accepted_mass_out > 0:
            w_ch4_out = self.methane_mass_fraction
            ch4_out = accepted_mass_out * w_ch4_out
            co2_out = accepted_mass_out * (1.0 - w_ch4_out)
            self.ch4_mass = max(0.0,self.ch4_mass - ch4_out,)
            self.co2_mass = max(0.0,self.co2_mass - co2_out,)

    def set_inherited_port_values(self, state: SimulationState):
        # First update values of the output port
        output_port = self.ports[self.output_port_name]
        output_port.methane_fraction = self.methane_fraction
        output_port.LHV = self.biogas_LHV
        output_port.density = self.biogas_density
        # Then create the output info object:
        output = {}
        output[self.output_port_name] = {
            'methane_fraction': self.methane_fraction,
            'LHV': self.biogas_LHV,
            'density': self.biogas_density
        }
        return output
