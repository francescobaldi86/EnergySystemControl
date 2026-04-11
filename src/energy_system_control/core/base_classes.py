from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class InitContext:
    environment: Any
    state: Any
    rng: Optional[Any] = None
    logger: Optional[Any] = None
    config: Optional[dict] = None


@dataclass
class EnvironmentalData:
    temperature_ambient: float = 293.15  # K
    temperature_cold_water: float = 288.15  # K
    direct_irradiation: float = 0.0      # W/m^2
    diffuse_irradiation: float = 0.0     # W/m^2
    solar_zenith: float | None = None           # degrees
    solar_azimuth: float | None = None         # degrees