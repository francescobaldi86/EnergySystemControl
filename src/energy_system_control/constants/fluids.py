# src/energy_system_control/constants/fluids.py
from __future__ import annotations
from dataclasses import dataclass
from .base import FrozenNamespace

@dataclass(frozen=True)
class Water(FrozenNamespace):
    # ~20°C, 1 atm 
    rho: float = 998.2     # density [kg·m⁻³]
    cp: float = 4.187     # [kJ·kg⁻¹·K⁻¹]
    k: float = 0.62856    # [W/mK] @ 40°C

@dataclass(frozen=True)
class Air(FrozenNamespace):
    # ~20°C, 1 atm
    rho: float = 1.2041    # [kg·m⁻³]
    cp: float = 1006.0     # [J·kg⁻¹·K⁻¹]

@dataclass(frozen=True)
class Methane(FrozenNamespace):
    LHV: float = 50_000.0  # [kJ/kg]
    rho: float = 0.72  # [kg/m3]
    MW : float = 16.04       # kg/kmol

@dataclass(frozen=True)
class CarbonDioxide(FrozenNamespace):
    rho = 1.98          # kg/m3
    MW : float = 44.01       # kg/kmol


WATER = Water()
AIR = Air()
METHANE = Methane()
CARBON_DIOXIDE = CarbonDioxide()
