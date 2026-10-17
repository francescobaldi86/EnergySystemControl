from energy_system_control.components.base import Grid
from energy_system_control.sim.state import SimulationState



class ElectricityGrid(Grid):
    cost_of_energy_purchased: float
    value_of_energy_sold: float
    def __init__(self, name: str, cost_of_electricity_purchased: float = 0.0, value_of_electricity_sold: float = 0.0):
        super().__init__(name, 'electricity')
        self.cost_of_energy_purchased = cost_of_electricity_purchased
        self.value_of_energy_sold = value_of_electricity_sold


class ColdWaterGrid(Grid):
    # Specific balancing utility for the cold water grid. Useful because it reads the temperature of the water
    def set_inherited_port_values(self, state: SimulationState):
        T_cold_water = state.environmental_data.temperature_cold_water
        self.ports[self.port_name].T = T_cold_water
        
        return {
            self.port_name: {
                'T': T_cold_water
            }
        }

class GasGrid(Grid):
    def __init__(self, name: str, utility_type: str = 'fuel'): 
        super().__init__(name, utility_type)

    def set_inherited_port_values(self, state: SimulationState):
        T_gas = state.environmental_data.temperature_cold_water
        self.ports[self.port_name].T = T_gas
        
        self.ports[self.port_name].LHV = 47000  # Esempio kJ/kg
        self.ports[self.port_name].density = 0.7  # Esempio kg/m3
        
        return {
            self.port_name: {
                'T': T_gas,
                'LHV': 47000,
                'density': 0.7
            }
        }