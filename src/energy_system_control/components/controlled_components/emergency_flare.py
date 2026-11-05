from energy_system_control.components.base import ControlledComponent
from energy_system_control.sim.state import SimulationState

class EmergencyFlare(ControlledComponent):
    """
    Emergency flare for excess biogas combustion.
    """
    def __init__(self, name: str, max_biogas_consumption_kW: float):
        self.max_biogas_consumption_kW = max_biogas_consumption_kW
        self.biogas_port_name = f'{name}_biogas_port'
        super().__init__(name=name, ports_info={self.biogas_port_name: 'biogas'})

    def step(self, state: SimulationState, action: float):
        if action is None: 
            return
            
        # action 1 means full flare operation, 0 means no operation
        # we limit the action to the range [0, 1]
        action = max(0.0, min(1.0, float(action)))
        consumed_power = self.max_biogas_consumption_kW * action
        
        port = self.ports[self.biogas_port_name]
        port.flows['chemical_energy'] = consumed_power  
        
        # Translation of chemical energy to mass and volume based on
        # the thermodynamic properties propagated by the BiogasStorage
        if getattr(port, 'LHV', 0.0) > 0:
            port.flows['mass'] = (consumed_power / port.LHV)
            port.flows['volume'] = (port.flows['mass'] / getattr(port, 'density', 1.0))
        else:
            port.flows['mass'] = 0.0
            port.flows['volume'] = 0.0