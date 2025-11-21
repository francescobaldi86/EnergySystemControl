# src/energy_system_control/constants/thermo.py
from __future__ import annotations
from dataclasses import dataclass
from .base import FrozenNamespace

@dataclass(frozen=True)
class Thermo(FrozenNamespace):
    sigma: float = 5.670374419e-8   # Stefan–Boltzmann constant [W·m⁻²·K⁻⁴] (CODATA 2018)
    R_univ: float = 8.314462618      # Universal gas constant [J·mol⁻¹·K⁻¹]
    g: float = 9.80665               # Standard gravity [m·s⁻²]

THERMO = Thermo()
