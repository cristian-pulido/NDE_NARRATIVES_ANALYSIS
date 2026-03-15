from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}


def read_tabular_file(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved, nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(resolved, nrows=nrows)
    raise ValueError(f"Unsupported tabular file format: {resolved}. Expected one of {sorted(SUPPORTED_TABULAR_SUFFIXES)}")
