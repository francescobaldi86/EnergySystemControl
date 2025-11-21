# src/energy_system_control/constants/base.py
from __future__ import annotations
from dataclasses import dataclass, replace
from contextlib import contextmanager
from typing import Any, Iterator

@dataclass(frozen=True)
class FrozenNamespace:
    """Immutable bag of constants. Values are plain floats (SI)."""

@contextmanager
def override(obj: Any, **updates: Any) -> Iterator[Any]:
    """
    Temporarily create a modified copy of a FrozenNamespace (or tree of them).
    Usage:
        with override(WATER, cp=4181.3) as W: ...
    """
    # Works for @dataclass(frozen=True)
    new = replace(obj, **updates)
    try:
        yield new
    finally:
        pass
