from energy_system_control.components.base import StorageUnit
from energy_system_control.helpers import *
from energy_system_control.core.base_classes import InitContext
from energy_system_control.constants import METHANE, CARBON_DIOXIDE
from energy_system_control.sim.state import SimulationState

class BiogasStorage(StorageUnit):

    def __init__(self, name: str, capacity: float, initial_ch4_mass: float = 0.0, initial_co2_mass: float = 0.0):
        self.capacity = capacity
        self.ch4_mass = initial_ch4_mass
        self.co2_mass = initial_co2_mass
        self.SOC_0 = self.total_mass / self.biogas_density / self.capacity

        self.input_port_name = f"{name}_input"
        self.output_port_name = f"{name}_output"

        super().__init__(name, {self.input_port_name: "fluid", self.output_port_name: "fluid"})

    def create_ports(self):
        ports = super().create_ports()
        
        # AGGIUNTO "volume" per intercettare la propagazione dal digestore
        custom_layers = ["mass", "volume", "chemical_energy"]
        
        for p_name in [self.input_port_name, self.output_port_name]:
            port = self.ports[p_name]
            
            if hasattr(port, "layers"):
                new_layers = list(port.layers)
                for layer in custom_layers:
                    if layer not in new_layers:
                        new_layers.append(layer)
                port.layers = new_layers
                
            for layer in custom_layers:
                port.flows[layer] = 0.0
                
        return ports

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

        # Helper per estrarre in modo sicuro i flussi, sostituendo i NoneType con 0.0
        def get_flow(port, layer):
            val = port.flows.get(layer)
            return val if val is not None else 0.0

        # 1. INGRESSO (Consideriamo solo la componente entrante, quindi positiva)
        mass_flow_in = max(0.0, get_flow(input_port, "mass"))  
        methane_fraction_in = getattr(input_port, "methane_fraction", 0.0)

        density_in = (methane_fraction_in * METHANE.rho + (1.0 - methane_fraction_in) * CARBON_DIOXIDE.rho)
        methane_mass_frac_in = (methane_fraction_in * METHANE.rho) / density_in if density_in > 0 else 0.0 
        
        delta_mass_in = mass_flow_in * dt
        ch4_in = delta_mass_in * methane_mass_frac_in
        co2_in = delta_mass_in - ch4_in

        self.ch4_mass += ch4_in
        self.co2_mass += co2_in

        # 2. USCITA (I flussi propagati in uscita sono negativi per convenzione)
        val_mass_out = get_flow(output_port, "mass")
        mass_flow_out = abs(min(0.0, val_mass_out))
        
        if mass_flow_out == 0.0:
            val_energy_out = get_flow(output_port, "chemical_energy")
            energy_flow_out = abs(min(0.0, val_energy_out))
            mass_flow_out = energy_flow_out / self.biogas_LHV if self.biogas_LHV > 0 else 0.0

        delta_mass_out = mass_flow_out * dt  

        if delta_mass_out > self.total_mass:
            delta_mass_out = self.total_mass

        if delta_mass_out > 0:
            w_ch4 = self.methane_mass_fraction
            self.ch4_mass -= delta_mass_out * w_ch4
            self.co2_mass -= delta_mass_out * (1.0 - w_ch4)

        self.SOC = self.total_mass / self.biogas_density / self.capacity

        # 3. SANITIZZAZIONE ANTI-NONE (Essenziale per il simulator.py originale)
        # Trasforma tutti i layer vuoti in 0.0 prima di restituire il controllo al simulatore
        for port in self.ports.values():
            for layer, val in port.flows.items():
                if val is None:
                    port.flows[layer] = 0.0

    def set_inherited_port_values(self, state: SimulationState):
        output_port = self.ports[self.output_port_name]
        output_port.methane_fraction = self.methane_fraction
        output_port.LHV = self.biogas_LHV
        output_port.density = self.biogas_density
        return [self.output_port_name]