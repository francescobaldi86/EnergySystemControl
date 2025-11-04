# tests/unit/test_io_loaders.py
from pathlib import Path
import pandas as pd

def test_csv_loader_parses_datetime(data_dir):
    from energy_system_control.io.loaders import load_csv_timeseries
    df = load_csv_timeseries(Path(data_dir) / "weather_small.csv")
    assert df.index.is_monotonic_increasing
    assert "ghi" in df.columns
