"""Deterministic and historical portfolio stress evaluation."""
import numpy as np
from numpy.typing import ArrayLike, NDArray


def stress_pnl(weights: ArrayLike, shocks: ArrayLike, *, notional: float = 1.0) -> NDArray[np.float64]:
    """Scenario P&L = notional * shocks @ weights; shocks are simple returns."""
    w, s = np.asarray(weights, float), np.asarray(shocks, float)
    if w.ndim != 1 or s.ndim != 2 or s.shape[1] != len(w) or not np.isfinite(s).all() or not np.isfinite(w).all():
        raise ValueError("incompatible stress inputs")
    return notional*(s @ w)


def historical_stress(returns: ArrayLike, weights: ArrayLike, *, notional: float = 1.0) -> tuple[float, int]:
    """Worst historical scenario loss=max(-notional*r[t] @ w), with scenario index."""
    pnl = stress_pnl(weights, returns, notional=notional)
    idx = int(np.argmin(pnl))
    return float(-pnl[idx]), idx
