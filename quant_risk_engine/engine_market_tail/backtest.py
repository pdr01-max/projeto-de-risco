"""Unconditional coverage and independence likelihood ratio backtests."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2
from scipy.special import xlogy


@dataclass(frozen=True)
class BacktestResult:
    statistic: float
    p_value: float
    breaches: int
    observations: int


def _ll(success: int, trials: int, probability: float) -> float:
    return float(xlogy(success, probability)+xlogy(trials-success, 1-probability))


def kupiec_pof(breaches: ArrayLike, alpha: float = 0.99) -> BacktestResult:
    """LR_POF=-2[log L(p=1-alpha)-log L(p=N_b/N)] ~ chi-square(1)."""
    b = np.asarray(breaches)
    if b.ndim != 1 or len(b) < 2 or not np.isin(b, [0, 1]).all() or not 0 < alpha < 1:
        raise ValueError("binary breaches and valid alpha required")
    k, n = int(b.sum()), len(b)
    lr = max(0.0, 2*(_ll(k, n, k/n)-_ll(k, n, 1-alpha)))
    return BacktestResult(lr, float(chi2.sf(lr, 1)), k, n)


def christoffersen_independence(breaches: ArrayLike) -> BacktestResult:
    """LR_IND=2(log L(pi01,pi11)-log L(pi)), Markov transition independence test."""
    b = np.asarray(breaches)
    if b.ndim != 1 or len(b) < 3 or not np.isin(b, [0, 1]).all():
        raise ValueError("at least three binary breaches required")
    previous, current = b[:-1], b[1:]
    n00 = int(((previous == 0)&(current == 0)).sum())
    n01 = int(((previous == 0)&(current == 1)).sum())
    n10 = int(((previous == 1)&(current == 0)).sum())
    n11 = int(((previous == 1)&(current == 1)).sum())
    p0 = n01/(n00+n01) if n00+n01 else 0
    p1 = n11/(n10+n11) if n10+n11 else 0
    p = (n01+n11)/(len(b)-1)
    lr = max(0.0, 2*(_ll(n01,n00+n01,p0)+_ll(n11,n10+n11,p1)-_ll(n01+n11,len(b)-1,p)))
    return BacktestResult(lr, float(chi2.sf(lr, 1)), int(b.sum()), len(b))


def backtest_var(realized_returns: ArrayLike, forecast_var: ArrayLike, alpha: float = 0.99) -> tuple[BacktestResult, BacktestResult]:
    """Breach at t iff -r[t]>VaR[t], with forecasts computed before observing r[t]."""
    r, v = np.asarray(realized_returns, float), np.asarray(forecast_var, float)
    if r.ndim != 1 or v.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("aligned finite realized returns and forecasts required")
    b = -r > v
    return kupiec_pof(b, alpha), christoffersen_independence(b)
