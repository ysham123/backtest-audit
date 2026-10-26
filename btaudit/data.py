"""Loader for the committed S&P composite monthly snapshot (Shiller data).

The snapshot ``data/sp500_shiller_monthly.csv`` is documented in ``data/DATA.md``. In the
file, values that are not yet available are written as ``0.0``; this loader turns them into
missing values. The ``Dividend`` column is an annualized dividend, so the monthly total
return is

    r_t = (P_t + D_t / 12) / P_{t-1} - 1,

the same construction as Shiller's total return price series. The return series ends at
the last month with a reported dividend.

The ``SP500`` price is a monthly average of daily closes for most of the sample, which
induces positive first-order autocorrelation in monthly returns even for a random walk
(Working 1960). See ``data/DATA.md`` and the real-data experiment for how this is handled.

The loader expects a source checkout of the repository (the data directory is not part of
the installed package).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "sp500_shiller_monthly.csv"
SNAPSHOT_SHA256 = "3a45dffabc414afc298c12159abc84c2875f88175ff3d4473ef7417e725ecac6"
SNAPSHOT_ROWS = 1869
PERIODS_PER_YEAR = 12


def sha256_of(path: Path) -> str:
    """Hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(
    path: Path = SNAPSHOT_PATH, expected_sha256: str | None = SNAPSHOT_SHA256
) -> pd.DataFrame:
    """Read the snapshot, verify its checksum and mark placeholder zeros as missing.

    Args:
        path: CSV file to read.
        expected_sha256: Required digest; ``None`` skips verification.

    Returns:
        DataFrame indexed by month-start ``Date`` with float columns.
    """
    if expected_sha256 is not None:
        actual = sha256_of(path)
        if actual != expected_sha256:
            raise ValueError(f"checksum mismatch for {path}: {actual} != {expected_sha256}")
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if frame["SP500"].le(0).any():
        raise ValueError("price column contains non-positive values")
    return frame.replace(0.0, np.nan)


def monthly_total_returns(frame: pd.DataFrame | None = None) -> pd.Series:
    """Monthly total returns ``(P_t + D_t / 12) / P_{t-1} - 1``.

    The series runs from the second month of the file to the last month with a reported
    dividend. Raises if a dividend is missing inside that span or months are not contiguous.
    """
    frame = load_snapshot() if frame is None else frame
    last = frame["Dividend"].last_valid_index()
    span = frame.loc[:last]
    if span["Dividend"].isna().any():
        raise ValueError("missing dividend inside the sample")
    expected = pd.date_range(span.index[0], span.index[-1], freq="MS")
    if not span.index.equals(expected):
        raise ValueError("months are not contiguous")
    price, dividend = span["SP500"], span["Dividend"]
    total = (price + dividend / 12.0) / price.shift(1) - 1.0
    return total.iloc[1:].rename("total_return")
