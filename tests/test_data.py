"""The committed S&P snapshot matches its documentation and is loaded correctly."""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btaudit import data

DATA_MD = data.DATA_DIR / "DATA.md"


def test_snapshot_checksum_matches_code_and_documentation() -> None:
    digest = data.sha256_of(data.SNAPSHOT_PATH)
    assert digest == data.SNAPSHOT_SHA256
    assert digest in DATA_MD.read_text()


def test_row_count_matches_documentation() -> None:
    frame = data.load_snapshot()
    assert len(frame) == data.SNAPSHOT_ROWS
    documented = re.search(r"\| Rows \| (\d+) \(excluding header\)", DATA_MD.read_text())
    assert documented is not None and int(documented.group(1)) == data.SNAPSHOT_ROWS


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    copy = tmp_path / "copy.csv"
    shutil.copy(data.SNAPSHOT_PATH, copy)
    with open(copy, "a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        data.load_snapshot(copy)


def test_first_total_return_hand_computed() -> None:
    returns = data.monthly_total_returns()
    # Jan 1871: P = 4.44; Feb 1871: P = 4.50, annualized dividend 0.26.
    assert returns.iloc[0] == pytest.approx((4.50 + 0.26 / 12) / 4.44 - 1)
    assert returns.index[0] == pd.Timestamp("1871-02-01")


def test_series_stops_at_last_reported_dividend() -> None:
    frame = data.load_snapshot()
    returns = data.monthly_total_returns(frame)
    assert returns.index[-1] == pd.Timestamp("2023-06-01")
    assert frame.loc["2023-07-01":, "Dividend"].isna().all()
    assert len(returns) == 1829
    assert np.all(np.isfinite(returns.to_numpy()))


def test_internal_missing_dividend_raises() -> None:
    frame = data.load_snapshot()
    frame.loc["1950-03-01", "Dividend"] = np.nan
    with pytest.raises(ValueError, match="missing dividend"):
        data.monthly_total_returns(frame)
