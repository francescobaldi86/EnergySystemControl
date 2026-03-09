from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class InitContext:
    environment: Any
    state: Any
    rng: Optional[Any] = None
    logger: Optional[Any] = None
    config: Optional[dict] = None


class Node:
    name: str
    time: float
    time_id: int
    layers: List[str]
    flow: Dict[str, float]
    def __init__(self, name, layers):
        self.name = name
        self.time = 0.0
        self.time_id = 0
        self.layers = layers
        self.reset_flow_data()  # Sets each 

    def balance(self, flows):
        raise NotImplementedError
    
    def reset_flow_data(self):
        self.flow = {name: 0.0 for name in self.layers}


class Sensor():
    name: str
    current_measurement: float
    def __init__(self, name):
        self.name = name

    def get_measurement(self):
        return self.current_measurement
    
    def initialize(self, ctx: InitContext):
        self.get_measurement(ctx.environment, ctx.state)

    def reset(self):
        self.current_measurement = None



@dataclass
class EnvironmentalData:
    temperature_ambient: float = 293.15  # K
    temperature_cold_water: float = 288.15  # K
    direct_irradiation: float = 0.0      # W/m^2
    diffuse_irradiation: float = 0.0     # W/m^2
    solar_zenith: float | None = None           # degrees
    solar_azimuth: float | None = None         # degrees