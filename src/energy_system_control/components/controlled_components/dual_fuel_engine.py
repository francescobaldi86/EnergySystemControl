from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext


class DualFuelEngine(ControlledComponent):
    """
    Motore a combustione interna bi-fuel biogas / GPL (introdotto per test_9).

    Ha la stessa struttura di InternalCombustionEngine (combustion_engine.py): l'azione e' il fattore
    di carico alpha = P_el / P_el_design compreso tra 0 e 1 e il rendimento elettrico segue la stessa
    curva polinomiale. La differenza e' che il motore ha due porte combustibile:
        - una porta 'biogas', collegata all'uscita dell'accumulo di biogas;
        - una porta 'fuel', collegata alla fornitura di GPL (LPGSupply).

    Regola di ripartizione del combustibile:
        1. il fabbisogno P_el / eta(alpha) viene coperto prima con il biogas presente nell'accumulo,
           fino a lasciare la riserva minima min_biogas_soc;
        2. sotto la riserva minima si usa il GPL, che copre la parte mancante del fabbisogno.
    Con min_biogas_soc pari alla soglia inferiore di spegnimento del controllore, il GPL interviene
    solo dopo che lo stato di carica del gasometro e' sceso sotto quella soglia, cioe' per completare
    il tempo minimo di accensione.

    La riserva minima evita anche un problema di BiogasStorage: se il prelievo supera la massa presente
    l'accumulo lo limita, ma il motore produrrebbe comunque tutta la potenza richiesta (energia dal nulla).

    Ipotesi: stesso rendimento elettrico per i due combustibili.
    Convenzione segni: flusso entrante (assorbito) = positivo, flusso uscente (erogato) = negativo.

    Parameters
    ----------
    name : str
        Nome del componente.
    P_el_design : float
        Potenza elettrica nominale [kW].
    biogas_storage_name : str
        Nome del BiogasStorage collegato alla porta biogas (serve per conoscere il biogas disponibile).
    min_biogas_soc : float, optional
        Riserva minima di biogas che il motore non preleva [-]. Defaults to 0.2.
    efficiency_el : float, optional
        Rendimento elettrico costante, usato solo se polynomial_efficiency=False.
    polynomial_efficiency : bool, optional
        Usa la curva di rendimento in funzione del fattore di carico. Defaults to True.
    """

    def __init__(self, name: str, P_el_design: float, biogas_storage_name: str,
                 min_biogas_soc: float = 0.2, efficiency_el: float | None = None,
                 polynomial_efficiency: bool = True):
        self.P_el_design = P_el_design
        self.biogas_storage_name = biogas_storage_name
        self.min_biogas_soc = min_biogas_soc
        self.efficiency_el = efficiency_el
        self.polynomial_efficiency = polynomial_efficiency

        # definizione delle porte
        self.electric_port_name = f'{name}_electricity_port'
        self.biogas_port_name = f'{name}_biogas_port'
        self.lpg_port_name = f'{name}_fuel_port'

        # grandezze dell'ultimo passo, utili per il post-processing
        self.last_fuel_power = 0.0
        self.last_biogas_power = 0.0
        self.last_lpg_power = 0.0

        # registrazione delle porte
        super().__init__(name=name,
                         ports_info={self.electric_port_name: 'electricity',
                                     self.biogas_port_name: 'biogas',
                                     self.lpg_port_name: 'fuel'})

    def initialize(self, ctx: InitContext):
        super().initialize(ctx)
        # riferimento all'accumulo di biogas, per leggerne massa, densita' e potere calorifico
        self.biogas_storage = ctx.environment.components[self.biogas_storage_name]

    def get_efficiency(self, load_factor: float) -> float:
        """
        Calcola l'efficienza elettrica del motore in funzione del fattore di carico (alfa),
        con la stessa curva di InternalCombustionEngine.
        """
        if load_factor <= 0.0:
            return 0.0
        if not self.polynomial_efficiency and self.efficiency_el is not None:
            return self.efficiency_el
        return -0.3167 * (load_factor ** 2) + 0.4752 * load_factor + 0.0064

    @property
    def optimal_load_factor(self) -> float:
        """Fattore di carico di massimo rendimento: vertice della parabola, 0.4752 / (2 * 0.3167) = 0.75."""
        return 0.4752 / (2 * 0.3167)

    def get_available_biogas_power(self, state: SimulationState) -> float:
        """Potenza chimica [kW] di biogas prelevabile nel passo senza scendere sotto la riserva minima."""
        st = self.biogas_storage
        reserve_mass = self.min_biogas_soc * st.capacity * st.biogas_density
        usable_mass = max(0.0, st.total_mass - reserve_mass)        # [kg]
        return usable_mass * st.biogas_LHV / state.time_step          # [kJ/s] = [kW]

    def step(self, state: SimulationState, action: float):
        '''
        Calcola i flussi per il time-step corrente.
        Convenzione segni: Flusso entrante (assorbito) = Positivo. Flusso uscente (erogato) = Negativo.
        '''
        if action is None:
            return

        if not (0.0 <= action <= 1.0):
            raise ValueError(
                f'The control input to the component {self.name} of type "DualFuelEngine" '
                f'should be a load factor between 0.0 and 1.0. {action} was provided at time step {state.time}'
            )

        generated_power = self.P_el_design * action
        fuel_power = generated_power / self.get_efficiency(action) if generated_power > 0.0 else 0.0

        # ripartizione del combustibile: prima il biogas disponibile, poi il GPL
        biogas_power = min(fuel_power, self.get_available_biogas_power(state))
        lpg_power = fuel_power - biogas_power

        # calcolo elettricita' prodotta
        self.ports[self.electric_port_name].flows['electricity'] = -generated_power

        # biogas consumato (proprieta' propagate dal BiogasStorage)
        bio = self.ports[self.biogas_port_name]
        bio.flows['chemical_energy'] = biogas_power
        bio.flows['mass'] = biogas_power / bio.LHV if (bio.LHV or 0.0) > 0 else 0.0
        bio.flows['volume'] = bio.flows['mass'] / bio.density if (bio.density or 0.0) > 0 else 0.0

        # GPL consumato (proprieta' propagate da LPGSupply). FuelPort ha solo i layer 'mass' e 'chemical_energy'
        lpg = self.ports[self.lpg_port_name]
        lpg.flows['chemical_energy'] = lpg_power
        lpg.flows['mass'] = lpg_power / lpg.LHV if (lpg.LHV or 0.0) > 0 else 0.0

        self.last_fuel_power = fuel_power
        self.last_biogas_power = biogas_power
        self.last_lpg_power = lpg_power

    @property
    def biogas_share(self) -> float:
        """Quota di biogas sul combustibile dell'ultimo passo [-]."""
        return self.last_biogas_power / self.last_fuel_power if self.last_fuel_power > 0 else 0.0
