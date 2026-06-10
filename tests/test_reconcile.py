"""Set de pruebas de reconciliación individual vs consolidado"""

from __future__ import annotations

from cost_forecast.data.reconcile import reconcile_all


def test_reconciliation_matches_and_extends(cfg):
    table = reconcile_all(cfg)
    assert set(table["serie"]) == {"Price_X", "Price_Y", "Price_Z"}
    # Coincidencia perfecta dentro de tolerancia y todas extienden el histórico
    assert (table["match_rate_%"] >= 99.9).all()
    assert table["extiende_historico"].all()
