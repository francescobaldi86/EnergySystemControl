from energy_system_control.core.base_classes import Sensor
from energy_system_control.components.storage import HotWaterStorage, MultiNodeHotWaterTank

class FlowTemperatureSensor(Sensor):
    port_name: str
    def __init__(self, name: str, port_name: str):
        super().__init__(name)
        self.port_name = port_name

    def get_measurement(self, environment):
        self.current_measurement = environment.ports[self.port_name].T
        return super().get_measurement()


class PowerSensor(Sensor):
    port_name: str
    def __init__(self, name, port_name):
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

    def get_measurement(self, environment):
        self.current_measurement = environment.ports[self.port_name].flow / environment.time_step
        return super().get_measurement()

    
class SOCSensor(Sensor):
    component_name: str
    def __init__(self, name, component_name):
        super().__init__(name)
        self.component_name = component_name

    def get_measurement(self, environment):
        self.current_measurement = environment.components[self.component_name].SOC
        return super().get_measurement()


class TankTemperatureSensor(Sensor):
    component_name: str
    sensor_height: float
    sensor_height_id: int
    def __init__(self, name: str, component_name: str, sensor_height: float | None = None):
        super().__init__(name)
        self.component_name = component_name
        self.sensor_height = sensor_height
        self.sensor_height_id = None

    def get_measurement(self, environment):
        if isinstance(environment.components[self.component_name], HotWaterStorage):
            self.current_measurement = environment.components[self.component_name].temperature
        elif isinstance(environment.components[self.component_name], MultiNodeHotWaterTank):
            if not self.sensor_height_id:
                self.sensor_height_id = environment.components[self.component_name].identify_layer_by_height(
                    height = self.sensor_height, 
                    default = environment.components[self.component_name].number_of_layers // 2 - 1, 
                    output_type = 'layer_id')
            self.current_measurement = environment.components[self.component_name].T_layer[self.sensor_height_id]
        return super().get_measurement()