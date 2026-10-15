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
    """Sensor that reads a value from a Home Assistant entity.

    The sensor is bound to a specific Home Assistant entity and retrieves its
    state through the environment's Home Assistant client. It can either read a
    numeric value as a float or keep the raw state representation, depending on
    the ``is_float`` flag.

    Parameters
    ----------
    name : str
        Name of the sensor.
    entity_id : str
        Unique entity identifier exposed by Home Assistant.
    is_float : bool, default=True
        If ``True``, the sensor reads the state as a float; otherwise it keeps
        the value in its safe string-based form.
    """
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
