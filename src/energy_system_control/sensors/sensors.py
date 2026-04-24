from energy_system_control.components.storage_units.thermal_storage import HotWaterStorage, MultiNodeHotWaterTank
from energy_system_control.core.base_classes import InitContext
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
    
    def initialize(self, ctx: InitContext):
        self.measure(ctx.environment, ctx.state)

    def reset(self):
        self.current_measurement = None


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
        if state.time_id == 0:
            self.current_measurement = 0.0  # The first power measurement is always 0
        else:
            self.current_measurement = environment.ports[self.port_name].flows[self.flow_type]
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
        mass_flow_kg = environment.ports[self.port_name].flows['mass']  # in kg
        # heat_flow_kJ = environment.ports[self.port_name].flows['heat']  # in kJ
        
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
    
    def __init__(self, name: str, source_sensor_name: str, lookback_time: float, n_samples: int):
        super().__init__(name)
        self.source_sensor_name = source_sensor_name
        self.lookback_time = lookback_time  # in hours
        self.n_samples = n_samples
        
        if lookback_time <= 0:
            raise ValueError("'lookback_time' must be positive")
        if n_samples <= 0:
            raise ValueError("'n_samples' must be positive")
        
        # Store measurements as (timestamp, value) tuples
        self.memory = deque()
        self.current_measurement = np.array([])
    
    def measure(self, environment, state):
        """
        Read the latest measurement from the source sensor, add it to memory,
        remove old measurements, and resample to return n_samples values.
        """
        # Get the current time from state (convert to hours if needed)
        current_time = state.time
        
        # Get the measurement from the source sensor
        source_sensor = environment.sensors.get(self.source_sensor_name)
        if source_sensor is None:
            raise ValueError(f"Source sensor '{self.source_sensor_name}' not found in environment")
        
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
