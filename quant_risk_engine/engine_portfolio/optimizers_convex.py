"""Long-only convex portfolio optimizers; returns are arithmetic per period."""
from dataclasses import dataclass
import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from quant_risk_engine.core.matrix import repair_psd


def _matrix(returns: pd.DataFrame | ArrayLike) -> tuple[np.ndarray, pd.Index]:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("finite T x N returns with T>=3 and N>=2 required")
    names = returns.columns if isinstance(returns, pd.DataFrame) else pd.Index([f"asset_{i}" for i in range(x.shape[1])])
    return x, names


def _weights(value: np.ndarray | None, names: pd.Index) -> pd.Series:
    if value is None or not np.isfinite(value).all():
        raise RuntimeError("solver failed to return finite weights")
    w = np.maximum(np.asarray(value).ravel(), 0)
    if w.sum() <= 0:
        raise RuntimeError("solver returned zero weights")
    return pd.Series(w/w.sum(), index=names, name="weight")


@dataclass(frozen=True)
class MeanVarianceOptimizer:
    """Maximize mu'w - (risk_aversion/2)w'Sigma w, subject to 1'w=1, 0<=w<=cap."""
    risk_aversion: float = 5.0
    max_weight: float = 1.0

    def optimize(self, returns: pd.DataFrame | ArrayLike | None = None, *, mean: ArrayLike | None = None, covariance: ArrayLike | None = None) -> pd.Series:
        if returns is None and (mean is None or covariance is None):
            raise ValueError("provide returns or both mean and covariance")
        if returns is not None:
            x, names = _matrix(returns)
            mu = x.mean(axis=0) if mean is None else np.asarray(mean, float)
            cov = np.cov(x, rowvar=False) if covariance is None else covariance
        else:
            mu = np.asarray(mean, float)
            names = pd.Index([f"asset_{i}" for i in range(len(mu))])
            cov = covariance
        sigma = repair_psd(cov)
        n = len(mu)
        if mu.shape != (n,) or sigma.shape != (n, n) or not np.isfinite(mu).all() or self.risk_aversion <= 0 or not 1/n <= self.max_weight <= 1:
            raise ValueError("invalid parameters or infeasible cap")
        w = cp.Variable(n)
        problem = cp.Problem(cp.Maximize(mu@w - self.risk_aversion/2*cp.quad_form(w, sigma)), [cp.sum(w)==1, w>=0, w<=self.max_weight])
        problem.solve()
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"optimization status: {problem.status}")
        return _weights(w.value, names)


@dataclass(frozen=True)
class MinCVaROptimizer:
    """Minimize eta + sum(max(-R_t w-eta,0))/(T*(1-alpha)) under full investment."""
    alpha: float = .95
    max_weight: float = 1.0

    def optimize(self, returns: pd.DataFrame | ArrayLike) -> pd.Series:
        x, names = _matrix(returns)
        t, n = x.shape
        if not 0 < self.alpha < 1 or not 1/n <= self.max_weight <= 1:
            raise ValueError("invalid alpha or infeasible cap")
        w, eta, excess = cp.Variable(n), cp.Variable(), cp.Variable(t, nonneg=True)
        objective = cp.Minimize(eta + cp.sum(excess)/(t*(1-self.alpha)))
        problem = cp.Problem(objective, [cp.sum(w)==1, w>=0, w<=self.max_weight, excess >= -x@w-eta])
        problem.solve()
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"optimization status: {problem.status}")
        return _weights(w.value, names)
