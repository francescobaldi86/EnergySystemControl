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
        
        # registrazione delle porte
        super().__init__(name=name,
                         ports_info={self.electric_port_name: 'electricity', self.fuel_port_name: 'fluid'})

    def create_ports(self):
        # 1. Creazione fisica delle porte tramite la classe genitore
        ports = super().create_ports()
        fuel_port = self.ports[self.fuel_port_name]
        
        # 2. Pre-registrazione dei layer che potrebbero arrivare da componenti connessi (es. BiogasStorage)
        custom_layers = ["mass", "volume", "chemical_energy"]
        
        if hasattr(fuel_port, "layers"):
            new_layers = list(fuel_port.layers)
            for layer in custom_layers:
                if layer not in new_layers:
                    new_layers.append(layer)
            fuel_port.layers = new_layers
            
        # 3. Inizializzazione sicura
        for layer in custom_layers:
            fuel_port.flows[layer] = 0.0
            
        return ports
        
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
        # calcolo combustibile consumato (espresso come potenza chimica entrante)
        self.ports[self.fuel_port_name].flows['chemical_energy'] = fuel_consumed