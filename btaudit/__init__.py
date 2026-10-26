"""btaudit: statistical tools for detecting backtest overfitting.

Modules
-------
sharpe      Sharpe ratio, Probabilistic and Deflated Sharpe Ratio, expected maximum Sharpe.
pbo         Probability of Backtest Overfitting via CSCV.
bootstrap   Stationary bootstrap, White's Reality Check, Hansen's SPA test.
multiple    Bonferroni, Holm and Benjamini-Hochberg corrections.
strategies  Rule-based timing strategies with lagged signals and transaction costs.
sims        Null and planted-alpha synthetic strategy universes.
data        Loader for the committed S&P composite monthly snapshot.
"""

__version__ = "1.0.0"
