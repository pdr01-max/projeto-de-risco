"""Copula dependence estimation and vectorized uniform simulation."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kendalltau, norm, rankdata, t
from quant_risk_engine.core.matrix import validate_correlation


@dataclass(frozen=True)
class CopulaFit:
    family: str
    correlation: NDArray[np.float64]
    degrees_of_freedom: float = 6.0
    clayton_theta: float = 0.0


def fit_copula(returns: ArrayLike, family: str = "student_t", degrees_of_freedom: float = 6) -> CopulaFit:
    """Estimate pseudo-observations rank/(T+1); elliptical correlation from transformed ranks; Clayton theta=2*tau/(1-tau)."""
    x = np.asarray(returns, float)
    if x.ndim != 2 or x.shape[0] < 30 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("finite T x N sample required")
    if family not in {"gaussian", "student_t", "clayton"} or degrees_of_freedom <= 2:
        raise ValueError("unsupported family or degrees of freedom")
    u = rankdata(x, axis=0)/(len(x)+1)
    if family == "clayton":
        if x.shape[1] != 2:
            raise ValueError("Clayton implementation is bivariate")
        tau = float(kendalltau(u[:, 0], u[:, 1]).statistic)
        if not 0 < tau < 1:
            raise ValueError("Clayton requires positive dependence")
        return CopulaFit(family, np.eye(2), clayton_theta=2*tau/(1-tau))
    z = norm.ppf(u) if family == "gaussian" else t.ppf(u, degrees_of_freedom)
    return CopulaFit(family, validate_correlation(np.corrcoef(z, rowvar=False)), degrees_of_freedom)


def simulate_copula(fit: CopulaFit, n: int, seed: int = 0) -> NDArray[np.float64]:
    """Simulate uniforms via Gaussian/t elliptical transform or Clayton gamma frailty."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    if fit.family == "clayton":
        theta = fit.clayton_theta
        if theta <= 0:
            raise ValueError("positive Clayton theta required")
        frailty = rng.gamma(1/theta, 1, size=(n, 1))
        return np.clip((1+rng.exponential(size=(n, 2))/frailty)**(-1/theta), 1e-12, 1-1e-12)
    corr = validate_correlation(fit.correlation)
    z = rng.standard_normal((n, len(corr))) @ np.linalg.cholesky(corr).T
    if fit.family == "student_t":
        z /= np.sqrt(rng.chisquare(fit.degrees_of_freedom, size=(n, 1))/fit.degrees_of_freedom)
        return np.clip(t.cdf(z, fit.degrees_of_freedom), 1e-12, 1-1e-12)
    if fit.family == "gaussian":
        return np.clip(norm.cdf(z), 1e-12, 1-1e-12)
    raise ValueError("unsupported copula family")
