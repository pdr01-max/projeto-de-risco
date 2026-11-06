"""Peaks over threshold estimation and tail risk evaluation."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import genpareto, kstest


@dataclass(frozen=True)
class POTFit:
    threshold: float
    shape: float
    scale: float
    exceedance_rate: float
    exceedances: int = 0
    observations: int = 0
    ks_p_value: float = 0.0


def fit_pot(losses: ArrayLike, threshold_quantile: float = 0.90) -> POTFit:
    """Fit GPD to excesses Y=L-u | L>u by maximum likelihood; p_u=N_u/N."""
    x = np.asarray(losses, float)
    if x.ndim != 1 or len(x) < 100 or not np.isfinite(x).all() or not 0.5 < threshold_quantile < 0.99:
        raise ValueError("invalid POT sample or threshold")
    u = float(np.quantile(x, threshold_quantile))
    excess = x[x > u] - u
    if len(excess) < 5:
        raise ValueError("at least 5 threshold exceedances required")
    shape, _, scale = genpareto.fit(excess, floc=0)
    if not np.isfinite([shape, scale]).all() or scale <= 0 or shape >= .95:
        raise ValueError("GPD tail has invalid or infinite-ES shape")
    p_value = float(kstest(excess, "genpareto", args=(shape, 0, scale)).pvalue)
    return POTFit(u, float(shape), float(scale), len(excess)/len(x),
                  len(excess), len(x), p_value)


def pot_var_es(fit: POTFit, alpha: float = 0.99) -> tuple[float, float]:
    """VaR=u+beta/xi*((p_u/(1-alpha))^xi-1); ES=(VaR+beta-xi*u)/(1-xi)."""
    if not 1-fit.exceedance_rate < alpha < 1 or fit.scale <= 0:
        raise ValueError("alpha must exceed threshold quantile")
    xi = fit.shape
    ratio = fit.exceedance_rate/(1-alpha)
    var = fit.threshold + (fit.scale*np.log(ratio) if abs(xi) < 1e-8 else fit.scale*np.expm1(xi*np.log(ratio))/xi)
    if xi >= .95:
        raise ValueError("shape must be below 0.95 for a stable finite ES")
    es = (var + fit.scale - xi*fit.threshold)/(1-xi)
    return float(var), float(es)


def pot_quantile(fit: POTFit, probabilities: ArrayLike, body_losses: ArrayLike) -> np.ndarray:
    """Semi-parametric inverse CDF: empirical body and GPD tail above 1-p_u."""
    p = np.asarray(probabilities, float)
    body = np.asarray(body_losses, float)
    if not np.isfinite(p).all() or not np.isfinite(body).all():
        raise ValueError("finite probabilities and body required")
    threshold_probability = 1 - fit.exceedance_rate
    clipped = np.clip(p, 1e-9, 1-1e-9)
    empirical = np.quantile(body, np.minimum(clipped, threshold_probability), method="linear")
    survival = (1-clipped)/fit.exceedance_rate
    tail = fit.threshold + genpareto.isf(np.clip(survival, 1e-12, 1),
                                         fit.shape, scale=fit.scale)
    return np.where(clipped > threshold_probability, tail, empirical)


def filtered_pot(returns: ArrayLike, volatility: ArrayLike,
                 next_volatility: float, threshold_quantile: float,
                 alpha: float) -> tuple[float, float, POTFit]:
    """McNeil-Frey: fit GPD to -r_t/sigma_t, then multiply tail quantiles by sigma_(t+1)."""
    x = np.asarray(returns, float)
    sigma = np.asarray(volatility, float)
    if x.shape != sigma.shape or (sigma <= 0).any() or next_volatility <= 0:
        raise ValueError("aligned positive volatility required")
    fit = fit_pot(-x/sigma, threshold_quantile)
    var, es = pot_var_es(fit, alpha)
    return var*next_volatility, es*next_volatility, fit
