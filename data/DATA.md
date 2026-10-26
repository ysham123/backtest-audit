# Data

## sp500_shiller_monthly.csv

| Field | Value |
| --- | --- |
| Source URL | https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv |
| Maintainer | datasets/s-and-p-500 on GitHub (Data Packages) |
| Original source | Robert Shiller, U.S. stock market data (http://www.econ.yale.edu/~shiller/data.htm), through 2023-06; later rows are S&P 500 prices from FRED |
| License | ODC-PDDL-1.0 (Open Data Commons Public Domain Dedication and License), as declared by the maintainer; credit to Robert Shiller for the underlying data |
| Download date | 2026-10-23 |
| Rows | 1869 (excluding header), monthly from 1871-01-01 to 2026-09-01 |
| SHA-256 | `3a45dffabc414afc298c12159abc84c2875f88175ff3d4473ef7417e725ecac6` |

Columns: `Date, SP500, Dividend, Earnings, Consumer Price Index, Long Interest Rate, Real Price, Real Dividend, Real Earnings, PE10`.

The file is committed unmodified. `btaudit.data.load_snapshot` verifies the checksum before
reading it.

## How returns are built

* Values that are not yet available are written as `0.0` in the file. The loader treats
  them as missing. `Dividend` and `Earnings` are missing after 2023-06; the CPI, interest
  rate and real columns after 2023-09; `PE10` before 1881. `SP500` is complete.
* `Dividend` is an annualized dividend per share (S&P four-quarter totals since 1926,
  Cowles annual data before 1926, both linearly interpolated to months), so one month of
  dividend income is `Dividend / 12`.
* Monthly total return: `r_t = (SP500_t + Dividend_t / 12) / SP500_{t-1} - 1`.
* The total return series runs from 1871-02 to 2023-06 (1829 months), the last month with
  a reported dividend. The 39 later months have prices but no dividends and are excluded
  rather than imputed.
* No risk-free rate is subtracted: the file has no short-term rate. Sharpe ratios in this
  repository are therefore ratios of total return to volatility, and cash earns zero.

## Known property: averaged prices

Shiller's documentation describes `SP500` as a monthly average of daily closing prices.
Returns computed from averaged prices are positively autocorrelated even when the
underlying daily prices follow a random walk (Working 1960 derives a first-order
autocorrelation of 0.25 in the limit). In this snapshot the first-order autocorrelation of
monthly total returns is about 0.27 and the second-order autocorrelation is about -0.01.

Consequence for backtests: a signal computed at the end of month `t - 1` overlaps with
half of the month-`t` average-to-average return, so a one-month signal lag still contains a
partial look-ahead. The real-data experiment therefore uses a two-month lag as its primary
specification and reports the one-month lag separately to show the size of the artifact.

Reference: Working, H. (1960). Note on the correlation of first differences of averages in
a random chain. Econometrica, 28(4), 916-918.
