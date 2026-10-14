from energy_system_control.sensors.sensors import Sensor
from energy_system_control.io.core import IOClient
from energy_system_control.io.home_assistant import HomeAssistantClient
from energy_system_control.core.base_classes import InitContext

class PhysicalSensor(Sensor):
    """
    A sensor instance that interrogates a real sensor
    """
    client: IOClient


class HomeAssistantSensor(PhysicalSensor):
    """Sensor that uses Home Assistant to get data"""
    entity_id: str
    is_float: bool
    def __init__(self, name: str, entity_id: str, is_float: bool = True):
        self.entity_id = entity_id
        self.is_float = is_float
        super().__init__(name)


    def initialize(self, context: InitContext):
        # Needed to connect to the client
        try:
            self.client = context.environment.clients['home_assistant']
        except KeyError:
            raise ValueError("Home Assistant client not found in the environment")


    def measure(self, environment, state):
        if self.is_float is True:
            self.current_measurement = self.client.get_float_state(self.entity_id)
            return self.current_measurement
        else:
            self.current_measurement = self.client.get_safe_state(self.entity_id)
            return self.current_measurement
