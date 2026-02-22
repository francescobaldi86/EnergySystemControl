from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
from abc import ABC, abstractmethod
import numpy as np

Mode = Literal["additive", "multiplicative"]

class UncertaintyModel(ABC):
    @abstractmethod
    def apply(self, value: float, *, rng: np.random.Generator) -> float:
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class NoUncertainty(UncertaintyModel):
    def apply(self, value: float, *, rng: np.random.Generator) -> float:
        return value


@dataclass(frozen=True, slots=True)
class GaussianUncertainty(UncertaintyModel):
    sigma: float                  # std dev (units depend on mode)
    mode: Mode = "additive"
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def apply(self, value: float, *, rng: np.random.Generator) -> float:
        eps = rng.normal(0.0, self.sigma)
        out = value * (1.0 + eps)

        if self.clip_min is not None:
            out = max(self.clip_min, out)
        if self.clip_max is not None:
            out = min(self.clip_max, out)
        return out


@dataclass(frozen=True, slots=True)
class UniformUncertainty(UncertaintyModel):
    half_width: float
    mode: Mode = "additive"
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def apply(self, value: float, *, rng: np.random.Generator) -> float:
        eps = rng.uniform(-self.half_width, self.half_width)
        out = value + eps if self.mode == "additive" else value * (1.0 + eps)
        if self.clip_min is not None:
            out = max(self.clip_min, out)
        if self.clip_max is not None:
            out = min(self.clip_max, out)
        return out
    
@dataclass(slots=True)
class AR1GaussianUncertainty(UncertaintyModel):
    sigma: float
    rho: float                    # 0..1
    mode: Mode = "additive"
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self):
        if not (0.0 <= self.rho < 1.0):
            raise ValueError("rho must be in [0,1).")
        self._eps_prev = 0.0

    def apply(self, value: float, *, rng: np.random.Generator) -> float:
        eps = self.rho * self._eps_prev + rng.normal(0.0, self.sigma)
        self._eps_prev = eps
        out = value + eps if self.mode == "additive" else value * (1.0 + eps)
        if self.clip_min is not None: out = max(self.clip_min, out)
        if self.clip_max is not None: out = min(self.clip_max, out)
        return out