"""Pure parameter estimators and chronological rolling forecasts."""
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import LedoitWolf
from .matrix import repair_psd


def estimate_moments(returns: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate mu=mean(r) and Sigma=sum((r-mu)(r-mu)')/(T-1)."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("finite T x N returns with T >= 3 required")
    return x.mean(axis=0), repair_psd(np.atleast_2d(np.cov(x, rowvar=False)))


def ledoit_wolf_covariance(returns: ArrayLike) -> NDArray[np.float64]:
    """Shrink sample covariance toward scaled identity using Ledoit-Wolf intensity."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("finite T x N returns with T >= 3 required")
    return repair_psd(LedoitWolf().fit(x).covariance_)


def rolling_forecast(returns: ArrayLike, window: int, estimator: Callable[[NDArray[np.float64]], float]) -> NDArray[np.float64]:
    """At date t estimate only on [t-window,t); output aligned to returns[window:]."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not 3 <= window < len(x):
        raise ValueError("invalid returns or rolling window")
    return np.fromiter((estimator(x[t-window:t]) for t in range(window, len(x))), dtype=float)
