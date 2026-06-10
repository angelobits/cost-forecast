"""Set de pruebas de los parsers por archivo"""

from __future__ import annotations

import pandas as pd

from cost_forecast.data.loaders import load_consolidated, load_x, load_y, load_z


def test_x_parser_sorts_ascending(cfg):
    s = load_x(cfg.path("raw_x"))
    assert s.is_monotonic_increasing is False or s.index.is_monotonic_increasing
    assert s.index.is_monotonic_increasing
    assert s.name == "Price_X"
    assert s.notna().all()


def test_y_parser_handles_bom_and_european_decimal(cfg):
    s = load_y(cfg.path("raw_y"))
    assert s.dtype.kind == "f"
    assert s.index.is_monotonic_increasing
    assert s.max() < 10_000


def test_z_parser_inverted_columns(cfg):
    s = load_z(cfg.path("raw_z"))
    assert s.name == "Price_Z"
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.is_monotonic_increasing


def test_consolidated_shape(cfg):
    df = load_consolidated(cfg.path("consolidated"))
    assert list(df.columns) == ["Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"]
    assert df.isna().sum().sum() == 0
    assert len(df) == 3530
