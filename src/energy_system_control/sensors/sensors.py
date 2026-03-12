from energy_system_control.core.base_classes import Sensor
from energy_system_control.components.storage import HotWaterStorage, MultiNodeHotWaterTank

class FlowTemperatureSensor(Sensor):
    port_name: str
    def __init__(self, name: str, port_name: str):
        super().__init__(name)
        self.port_name = port_name

    def measure(self, environment, state):
        self.current_measurement = environment.ports[self.port_name].T
        return self.current_measurement


class PowerSensor(Sensor):
    port_name: str
    flow_type: str
    def __init__(self, name, port_name, flow_type):
        """
        Model of sensor that measures the power flow at a specific port

        Parameters
        ----------
        name : str
            Name of the sensor
        port_name : str
            Name of the port it measures power from
    """
        super().__init__(name)
        self.port_name = port_name
        self.flow_type = flow_type

    def measure(self, environment, state):
        self.current_measurement = environment.ports[self.port_name].flow[self.flow_type] / state.time_step
        return self.current_measurement


class ElectricPowerSensor(PowerSensor):
    def __init__(self, name: str, port_name: str):
        super().__init__(name, port_name, 'electricity')


    
class SOCSensor(Sensor):
    component_name: str
    def __init__(self, name, component_name):
        super().__init__(name)
        self.component_name = component_name

    def measure(self, environment, state):
        self.current_measurement = environment.components[self.component_name].SOC
        return self.current_measurement


class TankTemperatureSensor(Sensor):
    component_name: str
    sensor_height: float
    sensor_height_id: int
    def __init__(self, name: str, component_name: str, sensor_height: float | None = None):
        super().__init__(name)
        self.component_name = component_name
        self.sensor_height = sensor_height
        self.sensor_height_id = None

    def measure(self, environment, state):
        if isinstance(environment.components[self.component_name], HotWaterStorage):
            self.current_measurement = environment.components[self.component_name].temperature
        elif isinstance(environment.components[self.component_name], MultiNodeHotWaterTank):
            if not self.sensor_height_id:
                self.sensor_height_id = environment.components[self.component_name].identify_layer_by_height(
                    height = self.sensor_height, 
                    default = environment.components[self.component_name].number_of_layers // 2 - 1, 
                    output_type = 'layer_id')
            self.current_measurement = environment.components[self.component_name].T_layer[self.sensor_height_id]
        return self.current_measurement


class HotWaterDemandSensor(Sensor):
    """
    Sensor that measures the net heat flow from a hot water demand component.
    
    This sensor reads the mass flow and temperature difference between the hot water
    supplied and the cold water return, and returns the net power in kW.
    The net heat flow is calculated as: Q_net = mdot * cp * (T_hot - T_cold)
    
    Parameters
    ----------
    name : str
        Name of the sensor
    component_name : str
        Name of the hot water demand component to measure
    """
    component_name: str
    port_name: str
    
    def __init__(self, name: str, component_name: str):
        super().__init__(name)
        self.component_name = component_name
        # The port name follows the pattern: {component_name}_fluid_port
        self.port_name = f'{component_name}_fluid_port'

    def measure(self, environment, state):
        from energy_system_control.constants import WATER
        
        # Get the mass flow and heat flow from the port
        mass_flow_kg = environment.ports[self.port_name].flow['mass']  # in kg
        heat_flow_kJ = environment.ports[self.port_name].flow['heat']  # in kJ
        
        # Get temperatures
        T_hot_water = environment.ports[self.port_name].T  # Hot water temperature in K
        T_cold_water = state.environmental_data.temperature_cold_water  # Cold water temperature in K
        
        # Handle case where temperature might not be set yet
        if T_hot_water is None or T_cold_water is None:
            self.current_measurement = 0.0
        else:
            # Calculate net heat flow: Q_net = mass_flow * cp * (T_hot - T_cold)
            # Q_net_kJ = mass_flow_kg * WATER.cp * (T_hot_water - T_cold_water)
            # Convert to power in kW
            Q_net_kJ = mass_flow_kg * WATER.cp * (T_hot_water - T_cold_water)
            self.current_measurement = Q_net_kJ / state.time_step
        
        return self.current_measurement