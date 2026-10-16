from typing import Literal

from energy_system_control.io.core import IOClient
from energy_system_control.core.base_classes import InitContext
from energy_system_control.constants.base import EPSILON

HA_ACTUATOR_DOMAIN_TYPES = Literal["switch", "light"]

class Actuator:
    """
    The actuator is a component that is used to interface the
    ESC library with the real world. It is used to send control signals
    to real systems
    """
    client: IOClient | None

    def __init__(self, name):
        self.name = name
        self.client = None



class HomeAssistantActuator(Actuator):
    entity_id: str
    domain: str
    service: str
    def __init__(self, 
                 name: str, 
                 entity_id: str):
        super().__init__(name)
        self.entity_id = entity_id
        self.domain = entity_id.split('.')[0]

    def initialize(self, context: InitContext):
        # Needed to connect to the client
        try:
            self.client = context.environment.clients['home_assistant']
        except KeyError:
            raise ValueError("Home Assistant client not found in the environment")

class HomeAssistantOnOffActuator(HomeAssistantActuator):
    """Actuator that toggles a Home Assistant binary entity on or off.

    This actuator wraps a Home Assistant entity such as a switch or light and
    maps a boolean control action to the corresponding HA service call.
    ``True`` sends ``turn_on`` and ``False`` sends ``turn_off``.

    Parameters
    ----------
    name : str
        Name of the actuator.
    entity_id : str
        Home Assistant entity identifier, for example ``switch.kitchen_light``.
    """
    service_on: str
    service_off: str
    def __init__(self, name, entity_id, on: str = "turn_on", off: str = "turn_off"):
        super().__init__(name, entity_id)
        self.service_on = on
        self.service_off = off

    def send_control_signal(self, state, action):
        if action == True:
            service = self.service_on
        elif action == False:
            service = self.service_off
        else:
            raise ValueError(f'This is a binary actuator, the action must be either True or False')
        self.client.call_service(
            domain = self.domain,
            service = service,
            entity_id = self.entity_id)

class HomeAssistantSmoothActuator(HomeAssistantActuator):
    """Actuator that controls a Home Assistant device with a continuous setpoint.

    This actuator is intended for entities that support a variable parameter,
    such as a dimmable light or a thermostat-like device. The provided action is
    converted to a float and sent to Home Assistant using the configured
    ``setpoint_varname`` field. When the action is effectively zero or below the
    configured epsilon, the actuator turns the entity off; otherwise it turns it
    on and passes the target value as a service parameter.

    Parameters
    ----------
    name : str
        Name of the actuator.
    entity_id : str
        Home Assistant entity identifier that will receive the command.
    setpoint_varname : str
        Name of the service parameter used to pass the numeric setpoint value to
        Home Assistant.
    """
    setpoint_varname: str
    service_on: str
    service_off: str
    def __init__(self, 
                 name: str, 
                 entity_id: str,
                 setpoint_varname: str,
                 on: str | None = "turn_on",
                 off: str | None = "turn_off"):
        super().__init__(name, entity_id)
        self.setpoint_varname = setpoint_varname
        self.service_on = on
        self.service_off = off
    
    def send_control_signal(self, state, action):
        try:
            action = float(action)
        except ValueError:
            raise ValueError(f'The action should be a float or a value that can be converted to a float. {action} was given instead')
        if action <= EPSILON:
            if self.service_on is not None:
                service = self.service_off
                other_info = {}
            else:
                service = self.setpoint_varname
                other_info = {'value': action}
        else:
            if self.service_on is not None:
                service = self.service_on
                other_info = {self.setpoint_varname: action}
            else:
                service = self.setpoint_varname
                other_info = {'value': action}
        # Call IO Client
        self.client.call_service(
            domain = self.domain,
            service = service,
            entity_id = self.entity_id,
            other_info = other_info
            )