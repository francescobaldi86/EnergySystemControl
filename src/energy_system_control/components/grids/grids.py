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
    def set_inherited_fluid_port_values(self, state: SimulationState):
        self.ports[self.port_name].T = state.environmental_data.temperature_cold_water
        return self.port_name, state.environmental_data.temperature_cold_water