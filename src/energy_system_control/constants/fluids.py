# src/energy_system_control/constants/fluids.py
from __future__ import annotations
from dataclasses import dataclass
from .base import FrozenNamespace

@dataclass(frozen=True)
class Water(FrozenNamespace):
    # Reference ~20°C, 1 atm (use CoolProp if you need T,p dependence)
    rho: float = 998.2     # density [kg·m⁻³]
    cp: float = 4.187     # specific heat [kJ·kg⁻¹·K⁻¹]
    k: float = 0.62856    # [W/mK] @ 40°C

@dataclass(frozen=True)
class Air(FrozenNamespace):
    # Dry air, ~20°C, 1 atm
    rho: float = 1.2041    # [kg·m⁻³]
    cp: float = 1006.0     # [J·kg⁻¹·K⁻¹]

WATER = Water()
AIR = Air()
