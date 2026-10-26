"""Experiment helpers, and consistency of the README with the committed result files."""

from pathlib import Path

import numpy as np
import pytest

from experiments import report
from experiments.common import ROOT, binomial_ci, rate_summary
from experiments.e3_power import oracle_power


def test_clopper_pearson_known_values() -> None:
    low, high = binomial_ci(0, 10)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** (1 / 10), rel=1e-9)
    low, high = binomial_ci(50, 1000)
    assert low < 0.05 < high
    assert rate_summary(50, 1000)["rate"] == 0.05


def test_oracle_power_matches_simulation() -> None:
    rng = np.random.default_rng(0)
    n_obs, sims = 252, 40_000
    x = rng.normal(1.0 / np.sqrt(252), 1.0, size=(sims, n_obs))
    t_stat = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n_obs))
    from scipy import stats

    simulated = np.mean(t_stat > stats.t.ppf(0.95, n_obs - 1))
    assert oracle_power(1.0, n_obs=n_obs) == pytest.approx(simulated, abs=0.01)


def test_p_value_formatting() -> None:
    assert report.p_value(0.0, n_boot=2000) == "< 0.0005"
    assert report.p_value(0.4391) == "0.439"
    assert report.p_value(0.0) == "< 1e-15"
    assert report.p_value(1.94e-7) == "1.9e-07"
    assert report.rule_label("ma_crossover(4,8)|LF|raw") == "ma_crossover(4,8), long/flat"
    assert report.rule_label("tsmom(5)|LS|VT") == "tsmom(5), long/short, vol-targeted"


def test_readme_tables_match_result_files() -> None:
    assert report.readme_is_current(), "run: python -m experiments.report --write"


DASHES = (chr(0x2013), chr(0x2014))  # en dash, em dash


def _text_files() -> list[Path]:
    suffixes = {".py", ".md", ".toml", ".yml", ".txt", ".json", ".cfg"}
    skip = {".git", ".pytest_cache", "__pycache__"}
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in suffixes]
    files += [ROOT / "Makefile", ROOT / "LICENSE"]
    return [p for p in files if not skip.intersection(p.parts)]


def test_no_en_or_em_dashes_in_repository() -> None:
    offenders = [
        str(p.relative_to(ROOT))
        for p in _text_files()
        if any(dash in p.read_text() for dash in DASHES)
    ]
    assert offenders == []
