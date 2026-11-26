from energy_system_control.components.storage_units.biogas_storage import BiogasStorage
from energy_system_control.sim.state import SimulationState


class BiogasStorageWithVent(BiogasStorage):
    """
    Accumulo di biogas (gasometro) con valvola di sfiato a solo traboccamento (introdotto per test_9).

    SOSTITUISCE (solo per test_9) il comportamento di BiogasStorage ad accumulo pieno, in
    biogas_storage.py:
        - BiogasStorage._update_soc limita lo stato di carica a 1.0, ma BiogasStorage.step continua
          ad aggiungere la massa in ingresso anche oltre la capacita'. Il gasometro diventa cosi' un
          serbatoio illimitato e il biogas in eccesso non viene ne' rilasciato ne' contato.
        - Qui, dopo il bilancio di massa di BiogasStorage.step, il volume che supera la capacita'
          viene tolto dall'accumulo (con la composizione attuale del gas) e mandato sulla porta
          '<name>_flare_output' come biogas sfiatato. Lo stato di carica non supera mai 1.0.

    La porta '<name>_flare_output' va collegata a un AtmosphericVent. La torcia (EmergencyFlare) e il
    suo controllore (ModulatingFlareController) non servono piu': lo sfiato non e' una scelta del
    controllore ma la conseguenza fisica del gasometro pieno, e si apre solo per la quantita' in eccesso.

    Parameters
    ----------
    Gli stessi di BiogasStorage (capacity in m3).
    """

    def __init__(self, name: str, capacity: float, initial_soc: float = 0.0, initial_methane_fraction: float = 0.5):
        super().__init__(name, capacity, initial_soc, initial_methane_fraction)
        self.vented_mass_total = 0.0        # [kg] biogas sfiatato dall'inizio della simulazione
        self.vented_ch4_mass_total = 0.0    # [kg] metano sfiatato dall'inizio della simulazione

    def step(self, state: SimulationState, action=None):
        # 1. Bilancio di massa originale (ingresso dal digestore, prelievo del motore)
        super().step(state, action)

        # 2. Traboccamento: volume oltre la capacita' del gasometro
        dt = state.time_step
        density = self.biogas_density
        excess_volume = max(0.0, self.volume_m3 - self.capacity)    # [m3]
        excess_mass = excess_volume * density                        # [kg]
        w_ch4 = self.methane_mass_fraction
        lhv = self.biogas_LHV

        if excess_mass > 0.0:
            self.ch4_mass -= excess_mass * w_ch4
            self.co2_mass -= excess_mass * (1.0 - w_ch4)
            self.vented_mass_total += excess_mass
            self.vented_ch4_mass_total += excess_mass * w_ch4
            self._update_soc()

        # 3. Flusso sulla porta di sfiato (uscente dall'accumulo = negativo)
        vent_port = self.ports.get(self.flare_port_name)
        if vent_port is not None:
            vent_port.flows['mass'] = -excess_mass / dt
            vent_port.flows['volume'] = -excess_volume / dt
            vent_port.flows['chemical_energy'] = -excess_mass / dt * lhv
