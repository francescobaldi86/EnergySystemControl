from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class SignalKey:
    main_key: str
    secondary_key: str      # e.g. "electric", "heat"

class SignalRegistry:
    _key_to_col: Dict[SignalKey, int]
    _col_to_key: List[SignalKey]
    
    def __init__(self):
        self._key_to_col = {}
        self._col_to_key = []

    def register(self, main_key: str, secondary_key: str) -> int:
        key = SignalKey(main_key, secondary_key)
        col = len(self._col_to_key)
        self._key_to_col[key] = col
        self._col_to_key.append(key)
        return col

    def col_index(self, main_key: str, secondary_key: str) -> int:
        return self._key_to_col[SignalKey(main_key, secondary_key)]