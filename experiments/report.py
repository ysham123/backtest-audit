"""Render the README result tables from ``results/*.json``.

Every number in the README tables is produced here from the committed result files.

Usage:
    python -m experiments.report --write   # rewrite the generated sections of README.md
    python -m experiments.report --check   # exit with status 1 if README.md is stale
"""

from __future__ import annotations

import argparse
import re
import sys

from experiments.common import ROOT, load_result
from experiments.methods import LABELS

README = ROOT / "README.md"
CORRECTED = ("dsr", "reality_check", "spa", "bonferroni", "holm", "bh")
DIAGNOSTIC = ("spa_iid",)
E2_COLUMNS = (
    ("normal_n1", "Normal, N=1"),
    ("normal_n10", "Normal, N=10"),
    ("normal_n100", "Normal, N=100"),
    ("normal_n1000", "Normal, N=1000"),
    ("t5_n100", "t(5), N=100"),
    ("garch_t5_n100", "GARCH t(5), N=100"),
    ("normal_rho05_n100", "Corr. 0.5, N=100"),
)
E3_METHODS = ("dsr", "reality_check", "spa", "holm", "bh")


def pct(rate: float, digits: int = 1) -> str:
    """Format a proportion as a percentage."""
    return f"{100 * rate:.{digits}f}%"


def pct_ci(summary: dict) -> str:
    """Rate with its 95% interval, e.g. ``4.6% [3.4, 6.1]``."""
    return f"{pct(summary['rate'])} [{100 * summary['ci_low']:.1f}, {100 * summary['ci_high']:.1f}]"


def p_value(p: float, n_boot: int | None = None) -> str:
    """Format a p-value; a bootstrap p-value of zero is reported as a bound."""
    if n_boot is not None and p == 0.0:
        return f"< {1 / n_boot:.4f}"
    if p == 0.0:  # an analytic p-value below double precision resolution of 1 - CDF
        return "< 1e-15"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


def rule_label(name: str) -> str:
    """``ma_crossover(4,8)|LF|raw`` to ``ma_crossover(4,8), long/flat``."""
    core, side, scaling = name.split("|")
    side_text = "long/flat" if side == "LF" else "long/short"
    return f"{core}, {side_text}" + (", vol-targeted" if scaling == "VT" else "")


def table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """A GitHub-flavored markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _e1_rows(e1: dict) -> list[tuple[str, str, str]]:
    sc = e1["scenarios"]
    last = {k: sc[k]["rows"][-1] for k in ("normal_1y", "normal_10y")}
    worst = max(s["max_abs_formula_error_vs_exact_n_ge_10"] for s in sc.values())
    return [
        (
            "Best in-sample Sharpe of 10,000 zero-skill strategies, 1 year of daily data (E1)",
            f"{last['normal_1y']['mc_mean']:.2f} (formula {last['normal_1y']['formula']:.2f})",
            "Search alone, over pure noise, produces a Sharpe ratio that would look exceptional.",
        ),
        (
            "Same search with 10 years of data (E1)",
            f"{last['normal_10y']['mc_mean']:.2f} (formula {last['normal_10y']['formula']:.2f})",
            "More data shrinks the inflation, roughly with the square root of the sample length, "
            "but does not remove it.",
        ),
        (
            "Expected-maximum formula vs exact normal value, N from 10 to 10,000 (E1)",
            f"within {pct(worst)}",
            "The Bailey and Lopez de Prado closed form is an accurate deflation benchmark.",
        ),
    ]


def _e2_rows(e2: dict) -> list[tuple[str, str, str]]:
    cfg = e2["configs"]

    def rates(method: str, ns: tuple[int, ...]) -> str:
        return " / ".join(pct(cfg[f"normal_n{n}"]["methods"][method]["rate"]) for n in ns)

    stable = ("reality_check", "bonferroni", "holm", "bh")
    worst = max(
        ((m, k) for k, _ in E2_COLUMNS for m in stable),
        key=lambda mk: cfg[mk[1]]["methods"][mk[0]]["rate"],
    )
    big = cfg["normal_n1000"]["methods"]
    return [
        (
            "False positive rate of a naive t-test on the best of N = 10 / 100 / 1000, "
            "nominal 5% (E2)",
            rates("naive_t", (10, 100, 1000)),
            "Testing the winner as if it were the only strategy tried is badly miscalibrated.",
        ),
        (
            "Highest false positive rate of Reality Check, Bonferroni, Holm or BH over "
            "7 null settings (E2)",
            f"{pct_ci(cfg[worst[1]]['methods'][worst[0]])}: {LABELS[worst[0]]}, "
            f"{dict(E2_COLUMNS)[worst[1]]}",
            "These corrections hold the 5% level within sampling error in every setting tried.",
        ),
        (
            "False positive rate of SPA (consistent) at N = 1 / 100 / 1000 (E2)",
            f"{rates('spa', (1, 100, 1000))} (iid-scaled diagnostic at N = 1000: "
            f"{pct(big['spa_iid']['rate'])})",
            "SPA over-rejects as N grows. The cause is studentizing by an estimated long-run "
            "variance whose noise the bootstrap ignores.",
        ),
        (
            "False positive rate of DSR at N = 10 / 100 / 1000 (E2)",
            rates("dsr", (10, 100, 1000)),
            "DSR is far more conservative than 5%: it compares the best Sharpe with the expected "
            "maximum, not with an upper quantile of the maximum.",
        ),
    ]


def _e3_rows(e3: dict) -> list[tuple[str, str, str]]:
    def cell(n: int, sr: float) -> str:
        c = e3["cells"][f"n{n}_sr{sr:g}"]
        rates = " / ".join(pct(c["detect"][m]["rate"], 0) for m in ("spa", "holm", "dsr"))
        return f"{rates} (oracle {pct(c['oracle_power'], 0)})"

    return [
        (
            "Detection rate, true Sharpe 1.0 among 100 strategies: SPA / Holm / DSR (E3)",
            cell(100, 1.0),
            "A genuinely good 5-year track record is usually missed once 100 variants were tried.",
        ),
        (
            "Detection rate, true Sharpe 2.0 among 1000 strategies: SPA / Holm / DSR (E3)",
            cell(1000, 2.0),
            "Strong signals survive even a large search; the price of searching is paid in power.",
        ),
    ]


def _e4_rows(e4: dict) -> list[tuple[str, str, str]]:
    noise = " / ".join(f"{e4['noise'][str(n)]['pbo']['mean']:.2f}" for n in (10, 100, 1000))
    strong = e4["planted"]["3"]["pbo"]
    null, drifted = e4["rules_on_null_markets"], e4["rules_on_drifted_null_markets"]

    def spread(v: dict) -> str:
        pbo = v["pbo"]
        return f"mean {pbo['mean']:.2f}, 5th to 95th pct. {pbo['q05']:.2f} to {pbo['q95']:.2f}"

    return [
        (
            "Mean PBO on pure-noise universes, N = 10 / 100 / 1000 (E4)",
            noise,
            "On noise the in-sample winner is a coin flip out of sample, as PBO should report.",
        ),
        (
            "Mean PBO with one strategy of true Sharpe 3 among 100 (E4)",
            f"{strong['mean']:.3f}",
            "A dominant real signal drives PBO to zero.",
        ),
        (
            f"PBO of the {null['n_rules']}-rule grid on {null['n_markets']} simulated markets with "
            "nothing predictable (E4)",
            spread(null),
            "Centered near one half, but a single history can land far from it.",
        ),
        (
            f"Same, with a constant drift of {drifted['monthly_drift']} per month added (E4)",
            spread(drifted),
            "Drift alone, with no timing skill, pushes PBO down: long/flat rules keep beating "
            "long/short ones.",
        ),
    ]


def _e5_rows(e5: dict) -> list[tuple[str, str, str]]:
    p, a = e5["primary_lag2"], e5["artifact_lag1"]
    best, bh = p["best_in_sample"], p["buy_and_hold"]
    dsr = p["dsr"]["vs_zero"]
    spa = p["spa_vs_buy_and_hold"]
    check = e5["averaged_random_walk_check"]
    n_boot = e5["protocol"]["spa_boot"]
    return [
        (
            f"Best in-sample rule of {p['n_rules']}, {p['evaluation_start'][:4]} to 1989, "
            "2-month lag, 10 bps (E5)",
            f"{rule_label(best['rule'])}: Sharpe {best['is_sharpe']:.2f} vs {bh['is_sharpe']:.2f} "
            "buy-and-hold",
            "The search finds a rule that looks better than the index in sample.",
        ),
        (
            "Same rule out of sample, 1990 to 2023-06 (E5)",
            f"Sharpe {best['oos_sharpe']:.2f} vs {bh['oos_sharpe']:.2f} buy-and-hold",
            "It stays slightly ahead of the index out of sample, which alone says nothing about "
            "significance.",
        ),
        (
            f"DSR p-value of the in-sample best, {dsr['n_trials']} trials (E5)",
            f"{p_value(dsr['dsr_pvalue'])} ({p_value(dsr['dsr_pvalue_effective'])} with "
            f"{dsr['effective_trials']:.1f} effective trials)",
            "Not significant with the raw trial count; the answer hinges on how correlated trials "
            "are counted, and both versions test against zero, which buy-and-hold already beats.",
        ),
        (
            "PBO, CSCV with S = 16 over 1876 to 2023 (E5)",
            f"{p['pbo']['pbo']:.3f} all rules; "
            f"{p['pbo']['by_side']['long_flat']:.3f} long/flat only; "
            f"{p['pbo']['by_side']['long_short']:.3f} long/short only",
            "Low PBO here says the winner beats a weak median rule, not that it beats the index.",
        ),
        (
            "SPA p-value, best of all rules vs buy-and-hold (E5)",
            f"{p_value(spa['raw']['spa_consistent_p'], n_boot)} (volatility-matched "
            f"{p_value(spa['vol_matched']['spa_consistent_p'], n_boot)})",
            "No rule beats buy-and-hold once the search is accounted for.",
        ),
        (
            "One-month lag instead of two on averaged prices (E5)",
            f"SPA p {p_value(a['spa_vs_buy_and_hold']['raw']['spa_consistent_p'], n_boot)} "
            f"on the S&P; rejects on {pct(check['spa_reject_lag1']['rate'], 0)} of "
            f"{check['n_markets']} averaged "
            f"random walks (2-month lag: {pct(check['spa_reject_lag2']['rate'], 0)})",
            "The apparent edge is an artifact of monthly averaging, reproduced on pure noise.",
        ),
    ]


def key_results_table() -> str:
    """The headline table: finding, number, interpretation."""
    rows = (
        _e1_rows(load_result("e1_selection_inflation"))
        + _e2_rows(load_result("e2_calibration"))
        + _e3_rows(load_result("e3_power"))
        + _e4_rows(load_result("e4_pbo_sanity"))
        + _e5_rows(load_result("e5_real_data"))
    )
    return table(("Finding", "Number", "What it means"), rows)


def selection_table() -> str:
    """E1: Monte Carlo against formula and exact value for every scenario."""
    e1 = load_result("e1_selection_inflation")
    rows = []
    for sc in e1["scenarios"].values():
        top = sc["rows"][-1]
        rows.append(
            (
                sc["label"],
                f"{top['mc_mean']:.3f} +/- {top['mc_se']:.3f}",
                f"{top['formula']:.3f}",
                f"{top['exact_normal']:.3f}",
                f"{sc['fitted_over_theory']:.3f}",
                pct(sc["max_abs_formula_error_vs_exact_n_ge_10"]),
            )
        )
    header = (
        "Scenario",
        "Best of 10,000: Monte Carlo mean +/- s.e.",
        "Formula",
        "Exact (normal)",
        "Fitted / theoretical scale",
        "Max formula error vs exact, N >= 10",
    )
    return table(header, rows)


def pbo_table() -> str:
    """E4: distribution of PBO over universes with a known answer."""
    e4 = load_result("e4_pbo_sanity")
    groups = [(f"Noise, N = {k}", v) for k, v in e4["noise"].items()]
    groups += [
        (f"One skilled strategy among 100, true Sharpe {k}", v) for k, v in e4["planted"].items()
    ]
    for key, label in (
        ("rules_on_null_markets", "null market"),
        ("rules_on_drifted_null_markets", "null market with drift"),
    ):
        v = e4[key]
        groups.append((f"{v['n_rules']}-rule grid, {label}", v))
    rows = [
        (
            label,
            f"{v['pbo']['mean']:.3f}",
            f"{v['pbo']['median']:.3f}",
            f"{v['pbo']['q05']:.3f} to {v['pbo']['q95']:.3f}",
            pct(v["share_pbo_above_half"], 0),
        )
        for label, v in groups
    ]
    return table(
        ("Universe", "Mean PBO", "Median", "5th to 95th pct.", "Share with PBO > 0.5"), rows
    )


def calibration_table() -> str:
    """E2: false positive rates with 95% intervals for every method and null setting."""
    cfg = load_result("e2_calibration")["configs"]
    rows = [
        tuple([LABELS[m]] + [pct_ci(cfg[k]["methods"][m]) for k, _ in E2_COLUMNS])
        for m in ("naive_t", "naive_psr", *CORRECTED, *DIAGNOSTIC)
    ]
    return table(("Method", *[label for _, label in E2_COLUMNS]), rows)


def power_table() -> str:
    """E3: detection rates for every cell, with the oracle benchmark."""
    e3 = load_result("e3_power")
    rows = []
    for cell in e3["cells"].values():
        rows.append(
            (
                str(cell["n_strategies"]),
                f"{cell['annual_sharpe']:g}",
                pct(cell["oracle_power"]),
                *[pct(cell["detect"][m]["rate"]) for m in E3_METHODS],
            )
        )
    return table(("N", "True Sharpe", "Oracle", *[LABELS[m] for m in E3_METHODS]), rows)


def real_data_table() -> str:
    """E5: the primary two-month-lag audit next to the one-month-lag artifact run."""
    e5 = load_result("e5_real_data")
    n_boot = e5["protocol"]["spa_boot"]
    runs = (e5["primary_lag2"], e5["artifact_lag1"])

    def row(label: str, fn) -> tuple[str, ...]:
        return (label, *[fn(r) for r in runs])

    rows = [
        row(
            "Evaluation window",
            lambda r: f"{r['evaluation_start'][:7]} to {r['evaluation_end'][:7]}",
        ),
        row(
            "In-sample / out-of-sample months",
            lambda r: f"{r['n_months_is']} / {r['n_months_oos']}",
        ),
        row(
            "Buy-and-hold Sharpe, IS / OOS",
            lambda r: (
                f"{r['buy_and_hold']['is_sharpe']:.2f} / {r['buy_and_hold']['oos_sharpe']:.2f}"
            ),
        ),
        row("Best IS rule", lambda r: rule_label(r["best_in_sample"]["rule"])),
        row(
            "Best IS rule Sharpe, IS / OOS",
            lambda r: (
                f"{r['best_in_sample']['is_sharpe']:.2f} / {r['best_in_sample']['oos_sharpe']:.2f}"
            ),
        ),
        row(
            "Best IS rule OOS percentile among rules",
            lambda r: pct(r["best_in_sample"]["oos_percentile_among_rules"]),
        ),
        row(
            "Best IS rule turnover per year",
            lambda r: f"{r['best_in_sample']['annual_turnover']:.2f}",
        ),
        row(
            "log10 growth of 1 over the window, best IS rule / buy-and-hold",
            lambda r: (
                f"{r['best_in_sample']['log10_growth']:.2f} / "
                f"{r['buy_and_hold']['log10_growth']:.2f}"
            ),
        ),
        row(
            "Median rule Sharpe, IS / OOS",
            lambda r: f"{r['median_is_sharpe']:.2f} / {r['median_oos_sharpe']:.2f}",
        ),
        row(
            "Rules with OOS Sharpe above buy-and-hold",
            lambda r: pct(r["share_rules_oos_above_buy_and_hold"]),
        ),
        row(
            "Naive PSR p-value of best IS rule (vs zero)",
            lambda r: p_value(r["dsr"]["vs_zero"]["naive_psr_pvalue"]),
        ),
        row(
            "DSR benchmark SR0, 600 trials (annualized)",
            lambda r: f"{r['dsr']['vs_zero']['sr0_annual']:.2f}",
        ),
        row("DSR p-value, 600 trials", lambda r: p_value(r["dsr"]["vs_zero"]["dsr_pvalue"])),
        row(
            "Effective trials (eigenvalue participation ratio)",
            lambda r: f"{r['dsr']['vs_zero']['effective_trials']:.2f}",
        ),
        row(
            "DSR p-value, effective trials",
            lambda r: p_value(r["dsr"]["vs_zero"]["dsr_pvalue_effective"]),
        ),
        row(
            "Best IS active rule vs buy-and-hold",
            lambda r: rule_label(r["dsr"]["active_vs_buy_and_hold"]["rule"]),
        ),
        row(
            "Its IS information ratio / DSR p-value (600 trials)",
            lambda r: (
                f"{r['dsr']['active_vs_buy_and_hold']['sr_best_annual']:.2f} / "
                f"{p_value(r['dsr']['active_vs_buy_and_hold']['dsr_pvalue'])}"
            ),
        ),
        row("PBO, all rules", lambda r: f"{r['pbo']['pbo']:.3f}"),
        row(
            "PBO, long/flat rules / long/short rules",
            lambda r: (
                f"{r['pbo']['by_side']['long_flat']:.3f} / {r['pbo']['by_side']['long_short']:.3f}"
            ),
        ),
        row("CSCV probability of OOS loss", lambda r: f"{r['pbo']['prob_oos_loss']:.3f}"),
        row(
            "CSCV degradation slope (OOS on IS Sharpe)",
            lambda r: f"{r['pbo']['degradation_slope']:.2f}",
        ),
        row(
            "Reality Check p-value vs buy-and-hold",
            lambda r: p_value(r["spa_vs_buy_and_hold"]["raw"]["reality_check_p"], n_boot),
        ),
        row(
            "SPA p-value vs buy-and-hold",
            lambda r: p_value(r["spa_vs_buy_and_hold"]["raw"]["spa_consistent_p"], n_boot),
        ),
        row(
            "SPA p-value, volatility-matched",
            lambda r: p_value(r["spa_vs_buy_and_hold"]["vol_matched"]["spa_consistent_p"], n_boot),
        ),
        row(
            "SPA p-value, mean block 3 / 6 / 24 months",
            lambda r: " / ".join(
                p_value(s["spa_consistent_p"], n_boot)
                for s in r["spa_vs_buy_and_hold"]["raw_block_sensitivity"]
            ),
        ),
    ]
    return table(("Quantity", "2-month lag (primary)", "1-month lag (artifact check)"), rows)


def runtime_line() -> str:
    """Wall-clock time of the last full run, from ``results/runtime.json``."""
    runtime = load_result("runtime")
    parts = ", ".join(f"{name.split()[0]} {sec:.0f} s" for name, sec in runtime["seconds"].items())
    return (
        f"Last full run: {runtime['total_seconds'] / 60:.1f} minutes with "
        f"{runtime['workers']} worker processes ({parts})."
    )


SECTIONS = {
    "runtime": runtime_line,
    "key-results": key_results_table,
    "selection": selection_table,
    "calibration": calibration_table,
    "power": power_table,
    "pbo": pbo_table,
    "real-data": real_data_table,
}


def render(text: str) -> str:
    """Replace every ``<!-- BEGIN name -->...<!-- END name -->`` block with fresh output."""
    for name, build in SECTIONS.items():
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.DOTALL)
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"README is missing the {name} markers")
        text = text[: match.end(1)] + build() + "\n" + text[match.start(2) :]
    return text


def write_readme() -> None:
    """Rewrite the generated sections of README.md in place."""
    README.write_text(render(README.read_text()))


def readme_is_current() -> bool:
    """True when README.md matches the tables rendered from the result files."""
    text = README.read_text()
    return render(text) == text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_readme()
    elif not readme_is_current():
        print("README.md tables are out of date; run python -m experiments.report --write")
        sys.exit(1)
