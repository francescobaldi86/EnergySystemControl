from energy_system_control.core.base_classes import Sensor

class TemperatureSensor(Sensor):
    def __init__(self, name, node_name):
        super().__init__(name)
        self.node_name = node_name

    def get_measurement(self, environment):
        return environment.nodes[self.node_name].T
    
class PowerSensor(Sensor):
    """
    Reads the power flow to/from a component at a specific node
    :param: component_name      Name of the component
    :param: node_name           Name of the node
    """
    component_name: str
    node_name: str
    def __init__(self, name, component_name, node_name):
        super().__init__(name)
        self.component_name = component_name
        self.node_name = node_name

    def get_measurement(self, environment):
        return environment.nodes[self.node_name].flow[self.component_name]
    
class PowerBalanceSensor(Sensor):
    """
    Reads the power flow balance at a specific node
    :param: node_name           Name of the node
    """
    node_name: str
    def __init__(self, name, node_name):
        super().__init__(name)
        self.node_name = node_name

    def get_measurement(self, environment):
        return environment.nodes[self.node_name].delta
    
class SOCSensor(Sensor):
    component_name: str
    node_name: str
    def __init__(self, name, component_name, node_name):
        super().__init__(name)
        self.component_name = component_name
        self.node_name = node_name

    def get_measurement(self, environment):
        return environment.nodes[self.node_name].SOC