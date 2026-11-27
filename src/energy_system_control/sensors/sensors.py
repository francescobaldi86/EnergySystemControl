from energy_system_control.components.storage_units.thermal_storage import HotWaterStorage, MultiNodeHotWaterTank
from energy_system_control.core.base_classes import InitContext
from energy_system_control.core.port import Port
from abc import ABC, abstractmethod
from collections import deque
import numpy as np

class Sensor(ABC):
    name: str
    current_measurement: float
    def __init__(self, name):
        self.name = name

    def get_measurement(self):
        return self.current_measurement
    
    @abstractmethod
    def measure(self, environment, state):
        pass
    
    def _resolve_refs(self, environment):
        """Hook called once at initialization: resolve object references
        (ports, components, sensors) against the environment. Default: no-op."""
        pass

    def initialize(self, ctx: InitContext):
        self._resolve_refs(ctx.environment)
        self.measure(ctx.environment, ctx.state)

    def reset(self):
        self.current_measurement = None


class PortSensor(Sensor):
    """
    Used to unify functions related to sensors that operate on ports
    """
    port_name: str
    def __init__(self, name: str, port_name):
        super().__init__(name)
        self.port_spec = port_name
        self.port_name = port_name if isinstance(port_name, str) else getattr(port_name, 'name', None)
        self._port = port_name if isinstance(port_name, Port) else None

    def _resolve_refs(self, environment):
        resolver = getattr(environment, 'resolve_port', None)
        if resolver is not None:
            self._port = resolver(self.port_spec)
            self.port_name = self._port.name
    
    def _get_port(self, environment):
        return self._port if self._port is not None else environment.ports[self.port_name]


class ComponentSensor(Sensor):
    """
    Used to unify functions related to sensors that operate on components
    """
    component_name: str
    def __init__(self, name, component_name):
        # ``component_name`` may be a component name (str) or a Component object.
        super().__init__(name)
        self.component_spec = component_name
        self.component_name = component_name if isinstance(component_name, str) else component_name.name
        self._component = None if isinstance(component_name, str) else component_name

    def _resolve_refs(self, environment):
        resolver = getattr(environment, 'resolve_component', None)
        if resolver is not None:
            self._component = resolver(self.component_spec)
            self.component_name = self._component.name

    def _get_component(self, environment):
        if self._component is not None:
            return self._component
        if self.component_name not in environment.components:
            raise KeyError(f"Sensor '{self.name}': unknown component '{self.component_name}'. "
                            f"Available components: {sorted(environment.components)}")
        return environment.components[self.component_name]

class FlowTemperatureSensor(PortSensor):

    def measure(self, environment, state):
        self.current_measurement = self._get_port(environment).T
        return self.current_measurement


class PowerSensor(PortSensor):
    flow_type: str
    def __init__(self, name, port_name, flow_type, scale_factor: float = 1.0):
        """
        Model of sensor that measures the power flow at a specific port

        Parameters
        ----------
        name : str
            Name of the sensor
        port_name : Port | str | tuple | Component
            The port it measures power from: a Port object, a full port-name
            string, a (component, role) tuple, or a single-port component.
        flow_type : str
            The flow layer to measure (e.g. 'electricity', 'heat').
        """
        super().__init__(name, port_name)
        self.flow_type = flow_type
        self.scale_factor = scale_factor

    def measure(self, environment, state):
        if state.time_id == 0:
            self.current_measurement = 0.0  # The first power measurement is always 0
        else:
            self.current_measurement = self._get_port(environment).flows[self.flow_type]
        return self.current_measurement


class ElectricPowerSensor(PowerSensor):
    def __init__(self, name: str, port_name):
        super().__init__(name, port_name, 'electricity')

class HeatPowerSensor(PowerSensor):
    def __init__(self, name: str, port_name):
        super().__init__(name, port_name, 'heat')
    
class SOCSensor(ComponentSensor):
    

    def measure(self, environment, state):
        self.current_measurement = self._get_component(environment).SOC
        return self.current_measurement


class TankTemperatureSensor(ComponentSensor):
    sensor_height: float
    sensor_height_id: int
    def __init__(self, name: str, component_name, sensor_height: float | None = None):
        # ``component_name`` may be a component name (str) or a Component object.
        super().__init__(name, component_name)
        self.sensor_height = sensor_height
        self.sensor_height_id = None

    def measure(self, environment, state):
        component = self._get_component(environment)
        if isinstance(component, MultiNodeHotWaterTank):
            if not self.sensor_height_id:
                self.sensor_height_id = component.identify_layer_by_height(
                    height = self.sensor_height,
                    default = component.number_of_layers // 2 - 1,
                    output_type = 'layer_id')
            self.current_measurement = component.T_layer[self.sensor_height_id]
        elif isinstance(component, HotWaterStorage):
            self.current_measurement = component.temperature
        return self.current_measurement


class HotWaterDemandSensor(ComponentSensor):
    """
    Sensor that measures the net heat flow from a hot water demand component.
    
    This sensor reads the mass flow and temperature difference between the hot water
    supplied and the cold water return, and returns the net power in kW.
    The net heat flow is calculated as: Q_net = mdot * cp * (T_hot - T_cold)
    
    Parameters
    ----------
    name : str
        Name of the sensor
    component_name : str | Component
        The hot water demand component to measure (name or object).
    port_role : str
        The role of the demand's fluid port, default 'fluid_port'.
    """
    component_name: str
    port_name: str

    def __init__(self, name: str, component_name, port_role: str = 'fluid_port'):
        super().__init__(name, component_name)
        self.port_role = port_role
        # Legacy fallback name (pattern: {component_name}_fluid_port)
        self.port_name = f'{self.component_name}_fluid_port'
        self._port = None

    def _resolve_refs(self, environment):
        super()._resolve_refs(environment)
        resolve_port = getattr(environment, 'resolve_port', None)
        if resolve_port is not None:
            try:
                self._port = self._component.port(self.port_role)
            except (KeyError, AttributeError):
                self._port = resolve_port(f'{self.component_name}_fluid_port')
            self.port_name = self._port.name

    def _get_port(self, environment):
        return self._port if self._port is not None else environment.ports[self.port_name]

    def measure(self, environment, state):
        from energy_system_control.constants import WATER

        port = self._get_port(environment)
        # Get the mass flow and heat flow from the port
        mass_flow_kg = port.flows['mass']  # in kg
        # heat_flow_kJ = port.flows['heat']  # in kJ

        # Get temperatures
        T_hot_water = port.T  # Hot water temperature in K
        T_cold_water = state.environmental_data.temperature_cold_water  # Cold water temperature in K
        
        # Handle case where temperature might not be set yet
        if T_hot_water is None or T_cold_water is None:
            self.current_measurement = 0.0
        else:
            # Calculate net heat flow: Q_net = mass_flow * cp * (T_hot - T_cold)
            # Q_net_kJ = mass_flow_kg * WATER.cp * (T_hot_water - T_cold_water)
            # Convert to power in kW
            Q_net_kW = mass_flow_kg * WATER.cp * (T_hot_water - T_cold_water)
            self.current_measurement = Q_net_kW
        
        return self.current_measurement
    

class SensorWithMemory(Sensor):
    """
    A sensor that wraps another sensor and maintains a rolling buffer of its measurements.
    
    This sensor reads values from a source sensor and stores them for a specified time window.
    The stored measurements are resampled to a fixed number of values when accessed.
    
    Parameters
    ----------
    name : str
        Name of this sensor
    source_sensor_name : str
        Name of the sensor to read data from (must exist in environment.sensors)
    lookback_time : float
        How far back in time (in hours) to store measurements. 
        For example, lookback_time=12 stores 12 hours of data.
    n_samples : int
        Number of values to return in the current_measurement attribute.
        The stored measurements are resampled to this size.
    
    Attributes
    ----------
    current_measurement : np.ndarray
        Numpy array of n_samples values, resampled from the stored measurements
        in the lookback window, ordered chronologically.
    """
    
    source_sensor_name: str
    lookback_time: float
    n_samples: int
    
    def __init__(self, name: str, source_sensor_name, lookback_time: float, n_samples: int):
        # ``source_sensor_name`` may be a sensor name (str) or a Sensor object.
        super().__init__(name)
        self.source_sensor_spec = source_sensor_name
        self.source_sensor_name = source_sensor_name if isinstance(source_sensor_name, str) else source_sensor_name.name
        self._source_sensor = None if isinstance(source_sensor_name, str) else source_sensor_name
        self.lookback_time = lookback_time  # in hours
        self.n_samples = n_samples
        
        if lookback_time <= 0:
            raise ValueError("'lookback_time' must be positive")
        if n_samples <= 0:
            raise ValueError("'n_samples' must be positive")
        
        # Store measurements as (timestamp, value) tuples
        self.memory = deque()
        self.current_measurement = np.array([])

    def _resolve_refs(self, environment):
        resolver = getattr(environment, 'resolve_sensor', None)
        if resolver is not None:
            self._source_sensor = resolver(self.source_sensor_spec)
            self.source_sensor_name = self._source_sensor.name

    def measure(self, environment, state):
        """
        Read the latest measurement from the source sensor, add it to memory,
        remove old measurements, and resample to return n_samples values.
        """
        # Get the current time from state (convert to hours if needed)
        current_time = state.time
        
        # Get the measurement from the source sensor
        if self._source_sensor is not None:
            source_sensor = self._source_sensor
        else:
            source_sensor = environment.sensors.get(self.source_sensor_name)
        if source_sensor is None:
            raise ValueError(f"Sensor '{self.name}': source sensor '{self.source_sensor_name}' not found "
                             f"in environment. Available sensors: {sorted(environment.sensors)}")
        
        source_value = source_sensor.get_measurement()
        
        # Add the new measurement to memory
        self.memory.append((current_time, source_value))
        
        # Remove measurements older than lookback_time
        cutoff_time = current_time - self.lookback_time
        while self.memory and self.memory[0][0] < cutoff_time:
            self.memory.popleft()
        
        # Resample the memory to n_samples values
        self.current_measurement = self._resample_measurements()
        
        return self.current_measurement
    
    def _resample_measurements(self) -> np.ndarray:
        """
        Resample stored measurements to n_samples evenly distributed values.
        
        Returns
        -------
        np.ndarray
            Array of n_samples values interpolated from the stored measurements.
        """
        if len(self.memory) == 0:
            return np.array([])
        
        if len(self.memory) == 1:
            # If only one sample, return array of that value repeated n_samples times
            return np.full(self.n_samples, self.memory[0][1])
        
        # Extract times and values from memory
        times = np.array([t for t, v in self.memory])
        values = np.array([v for t, v in self.memory])
        
        # Create n_samples evenly spaced time points across the stored range
        target_times = np.linspace(times[0], times[-1], self.n_samples)
        
        # Interpolate values at the target times
        resampled_values = np.interp(target_times, times, values)
        
        return resampled_values
    
    def reset(self):
        """Reset the sensor and clear the memory buffer."""
        super().reset()
        self.memory.clear()
        self.current_measurement = np.array([])

