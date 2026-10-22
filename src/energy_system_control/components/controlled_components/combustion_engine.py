from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import FuelType

class InternalCombustionEngine(ControlledComponent):
    def __init__(self, name: str, P_el_design: float, efficiency_el: float | None = None, 
                 fuel: FuelType | None = None, polynomial_efficiency: bool = True):
        self.P_el_design = P_el_design
        self.efficiency_el = efficiency_el
        self.polynomial_efficiency = polynomial_efficiency

        # Riconoscimento del fuel (corretto l'operatore 'is' in '==')
        if fuel == 'generic_fuel':
            fuel = 'fuel'
        
        # definizione delle porte
        self.electric_port_name = f'{name}_electricity_port'
        self.fuel_port_name = f'{name}_{fuel}_port'
        
        # registrazione delle porte
        super().__init__(name=name,
                         ports_info={self.electric_port_name: 'electricity', self.fuel_port_name: fuel})

    def create_ports(self):
        ports = super().create_ports()
        return ports

    def get_efficiency(self, load_factor: float) -> float:
        """
        Calcola l'efficienza elettrica del motore in funzione del fattore di carico (alfa).
        Il parametro load_factor è il rapporto (P_el / P_el_design) ed è compreso tra 0.0 e 1.0.
        """
        if load_factor <= 0.0:
            return 0.0

        if not self.polynomial_efficiency and self.efficiency_el is not None:
            return self.efficiency_el

        eta = -0.3167 * (load_factor ** 2) + 0.4752 * load_factor + 0.0064
        return eta

    def step(self, state: SimulationState, action: float):
        '''
        Calcola i flussi per il time-step corrente.
        Convenzione segni: Flusso entrante (assorbito) = Positivo. Flusso uscente (erogato) = Negativo.
        '''
        if action is None:
            return

        # Sblocco per il controllo continuo: l'azione coincide con il fattore di carico
        if not (0.0 <= action <= 1.0):
            raise ValueError(
                f'The control input to the component {self.name} of type "InternalCombustionEngine" '
                f'should be a load factor between 0.0 and 1.0. {action} was provided at time step {state.time}'
            )

        generated_power = self.P_el_design * action

        if generated_power > 0.0:
            # L'action rappresenta esattamente il load_factor (alfa)
            current_efficiency = self.get_efficiency(action)
            fuel_consumed = generated_power / current_efficiency
        else:
            fuel_consumed = 0.0
        
        # calcolo elettricità prodotta
        self.ports[self.electric_port_name].flows['electricity'] = -generated_power
        # calcolo combustibile consumato (espresso come potenza chimica entrante)
        self.ports[self.fuel_port_name].flows['chemical_energy'] = fuel_consumed
        self.ports[self.fuel_port_name].flows['mass'] = fuel_consumed / self.ports[self.fuel_port_name].LHV
        self.ports[self.fuel_port_name].flows['volume'] = self.ports[self.fuel_port_name].flows['mass'] * self.ports[self.fuel_port_name].density