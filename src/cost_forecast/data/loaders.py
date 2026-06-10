"""Parsers robustos por archivo para las series de materias primas.

Cada CSV individual llega con un formato "sucio" distinto y deliberado:

* ``X.csv``  → ``Date,Price`` en ISO, **orden descendente** (2024 primero).
* ``Y.csv``  → **BOM** inicial, separador ``;``, decimal ``,``, fecha ``D/M/Y``.
* ``Z.csv``  → **columnas invertidas** ``Price,Date``.

La estrategia es un parser específico por archivo (alta cohesión) que normaliza
todo a un :class:`pandas.Series` indexado por fecha, ordenado y de-duplicado.
El consolidado limpio se carga aparte y es la **fuente de verdad**.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _finalize(series: pd.Series, name: str) -> pd.Series:
    """Ordena por fecha, elimina duplicados (último gana) y nombra la serie."""
    series = series.dropna()
    series = series[~series.index.duplicated(keep="last")]
    series = series.sort_index()
    series.name = name
    series.index.name = "Date"
    return series


def load_x(path: str | Path) -> pd.Series:
    """Carga ``X.csv`` (ISO, orden descendente). Devuelve Series ordenada asc."""
    df = pd.read_csv(path, dtype={"Price": "float64"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    s = df.set_index("Date")["Price"]
    logger.info("X.csv: %d filas crudas -> normalizadas", len(df))
    return _finalize(s, "Price_X")


def load_y(path: str | Path) -> pd.Series:
    """Carga ``Y.csv`` (BOM, sep ';', decimal ',', fecha D/M/Y)."""
    # ``utf-8-sig`` consume el BOM; el decimal europeo se maneja con ``decimal=','``.
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        dtype={"Price": "float64"},
    )
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    s = df.set_index("Date")["Price"]
    logger.info("Y.csv: %d filas crudas -> normalizadas", len(df))
    return _finalize(s, "Price_Y")


def load_z(path: str | Path) -> pd.Series:
    """Carga ``Z.csv`` (columnas invertidas ``Price,Date``)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    s = df.set_index("Date")["Price"]
    logger.info("Z.csv: %d filas crudas -> normalizadas", len(df))
    return _finalize(s, "Price_Z")


def load_consolidated(path: str | Path) -> pd.DataFrame:
    """Carga el consolidado limpio (fuente de verdad) indexado por fecha."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    logger.info("Consolidado: %d filas, columnas=%s", len(df), list(df.columns))
    return df


def load_all_individual(cfg: Config) -> dict[str, pd.Series]:
    """Carga las tres series individuales aplicando su parser específico."""
    return {
        "X": load_x(cfg.path("raw_x")),
        "Y": load_y(cfg.path("raw_y")),
        "Z": load_z(cfg.path("raw_z")),
    }
