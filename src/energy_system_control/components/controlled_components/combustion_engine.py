from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import OnOffComponentError

class InternalCombustionEngine(ControlledComponent):
    def __init__(self, name: str, P_el_design: float, efficiency_el: float):
        self.P_el_design = P_el_design
        self.efficiency_el = efficiency_el
        
        # definizione delle porte
        self.electric_port_name = f'{name}_electricity_port'
        self.fuel_port_name = f'{name}_fuel_port'
        
        # registrazione delle porte passando il dizionario ports_info per inizializzare la classe madre
        super().__init__(name=name,
                         ports_info={self.electric_port_name: 'electricity', self.fuel_port_name: 'fluid'})
        
    def step(self, state: SimulationState, action):
        """
        Calcola i flussi per il time-step corrente.
        Convenzione segni: Flusso entrante (assorbito) = Positivo. Flusso uscente (erogato) = Negativo.
        """
        for port in self.ports.values():
            for layer in port.flows.keys():
                port.flows[layer] = 0.0

        if action is None:
            return

        if action not in {0.0, 1.0}:
            raise OnOffComponentError(f'The control input to the component {self.name} of type "InternalCombustionEngine" should be either 1 or 0. {action} was provided at time step {state.time}')

        generated_power = self.P_el_design * action
        fuel_consumed = generated_power / self.efficiency_el
        # calcolo elettricità prodotta
        self.ports[self.electric_port_name].flows['electricity'] = -generated_power
        # calcolo combustibile consumato
        self.ports[self.fuel_port_name].flows['mass'] = fuel_consumed
        # self.ports[self.fuel_port_name].flows['heat'] = fuel_consumed * 50000