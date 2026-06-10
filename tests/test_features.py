"""Set de pruebas de estacionariedad, selección por votación y VIF"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_forecast.data.loaders import load_consolidated
from cost_forecast.features.selection import select_for_target
from cost_forecast.features.stationarity import assess_stationarity
from cost_forecast.features.transforms import difference
from cost_forecast.features.vif import compute_vif


def _inputs_targets(cfg):
    df = load_consolidated(cfg.path("consolidated")).rename(
        columns=lambda c: c.replace("Price_", "")
    )
    return df[["X", "Y", "Z"]], df


def test_levels_nonstationary_diffs_stationary(cfg):
    df = load_consolidated(cfg.path("consolidated"))
    lvl = assess_stationarity(df["Price_Equipo1"])
    dif = assess_stationarity(difference(df)["Price_Equipo1"])
    assert not lvl.adf_stationary  # nivel: no estacionario
    assert dif.adf_stationary  # diferencia: estacionario


def test_voting_recovers_data_driven_relationships(cfg):
    inputs, df = _inputs_targets(cfg)
    r1 = select_for_target(inputs, df["Equipo1"], cfg)
    r2 = select_for_target(inputs, df["Equipo2"], cfg)
    assert "Y" in r1.selected and "Z" not in r1.selected
    assert "Z" in r2.selected and "X" not in r2.selected


def test_partial_correlation_reveals_spurious_z(cfg):
    from cost_forecast.features.relationships import partial_correlation_table

    df = load_consolidated(cfg.path("consolidated")).rename(
        columns=lambda c: c.replace("Price_", "")
    )
    tbl = partial_correlation_table(df, ["X", "Y", "Z"], ["Equipo1", "Equipo2"])
    z_e1 = tbl[(tbl["equipo"] == "Equipo1") & (tbl["insumo"] == "Z")].iloc[0]
    y_e1 = tbl[(tbl["equipo"] == "Equipo1") & (tbl["insumo"] == "Y")].iloc[0]
    assert z_e1["corr_bivariada"] > 0.7 and abs(z_e1["corr_parcial"]) < 0.1  # espuria
    assert bool(z_e1["espuria"])
    assert abs(y_e1["corr_parcial"]) > 0.8  # Y sí es genuina


def test_vif_single_variable_is_one():
    df = pd.DataFrame({"a": np.arange(50.0)})
    vif = compute_vif(df)
    assert vif.loc[0, "vif"] == 1.0
