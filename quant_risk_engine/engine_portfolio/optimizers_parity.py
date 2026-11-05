"""Equal risk contribution and hierarchical risk parity."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from quant_risk_engine.core.matrix import repair_psd, validate_correlation


def _covariance(returns: pd.DataFrame | ArrayLike | None, covariance: ArrayLike | None) -> tuple[np.ndarray, pd.Index]:
    if returns is None and covariance is None:
        raise ValueError("returns or covariance required")
    if returns is not None:
        x = np.asarray(returns, float)
        if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
            raise ValueError("finite T x N returns required")
        names = returns.columns if isinstance(returns, pd.DataFrame) else pd.Index([f"asset_{i}" for i in range(x.shape[1])])
        cov = np.cov(x, rowvar=False) if covariance is None else covariance
    else:
        cov = covariance
        names = pd.Index([f"asset_{i}" for i in range(np.asarray(cov).shape[0])])
    sigma = repair_psd(cov)
    if len(sigma) < 2 or sigma.shape != (len(names), len(names)):
        raise ValueError("covariance dimensions mismatch")
    return sigma, names


@dataclass(frozen=True)
class RiskParityOptimizer:
    """Find long-only w with equal risk contributions RC_i=w_i(Sigma w)_i/sqrt(w'Sigma w)."""
    tolerance: float = 1e-8

    def optimize(self, returns: pd.DataFrame | ArrayLike | None = None, *, covariance: ArrayLike | None = None) -> pd.Series:
        sigma, names = _covariance(returns, covariance)
        n = len(names)
        def objective(w: np.ndarray) -> float:
            contribution = w*(sigma@w)
            return float(np.sum((contribution-contribution.sum()/n)**2)/(contribution.sum()**2))
        result = minimize(objective, np.full(n, 1/n), method="SLSQP", bounds=[(1e-8, 1)]*n, constraints={"type":"eq", "fun":lambda w: w.sum()-1}, options={"ftol":self.tolerance,"maxiter":2000})
        if not result.success:
            raise RuntimeError(f"risk parity optimization failed: {result.message}")
        w = np.maximum(result.x, 0)
        return pd.Series(w/w.sum(), index=names, name="weight")


@dataclass(frozen=True)
class HRPOptimizer:
    """López de Prado HRP: cluster on d_ij=sqrt((1-rho_ij)/2), then recursive bisection."""
    linkage_method: str = "single"

    def optimize(self, returns: pd.DataFrame | ArrayLike | None = None, *, covariance: ArrayLike | None = None) -> pd.Series:
        sigma, names = _covariance(returns, covariance)
        vol = np.sqrt(np.diag(sigma))
        corr = validate_correlation(sigma/np.outer(vol, vol))
        distance = np.sqrt(np.maximum((1-corr)/2, 0))
        tree = linkage(squareform(distance, checks=False), method=self.linkage_method)
        order = leaves_list(tree).tolist()
        weights = np.ones(len(names))

        def cluster_variance(indices: list[int]) -> float:
            sub = sigma[np.ix_(indices, indices)]
            inverse_diag = 1/np.diag(sub)
            ivp = inverse_diag/inverse_diag.sum()
            return float(ivp@sub@ivp)

        clusters = [order]
        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                split = len(cluster)//2
                left, right = cluster[:split], cluster[split:]
                lv, rv = cluster_variance(left), cluster_variance(right)
                allocation = rv/(lv+rv)
                weights[left] *= allocation
                weights[right] *= 1-allocation
                next_clusters.extend((left, right))
            clusters = next_clusters
        return pd.Series(weights/weights.sum(), index=names, name="weight")
