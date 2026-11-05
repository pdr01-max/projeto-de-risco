"""Peaks over threshold estimation and tail risk evaluation."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import genpareto


@dataclass(frozen=True)
class POTFit:
    threshold: float
    shape: float
    scale: float
    exceedance_rate: float


def fit_pot(losses: ArrayLike, threshold_quantile: float = 0.90) -> POTFit:
    """Fit GPD to excesses Y=L-u | L>u by maximum likelihood; p_u=N_u/N."""
    x = np.asarray(losses, float)
    if x.ndim != 1 or len(x) < 100 or not np.isfinite(x).all() or not 0.5 < threshold_quantile < 0.99:
        raise ValueError("invalid POT sample or threshold")
    u = float(np.quantile(x, threshold_quantile))
    excess = x[x > u] - u
    if len(excess) < 20:
        raise ValueError("at least 20 threshold exceedances required")
    shape, _, scale = genpareto.fit(excess, floc=0)
    return POTFit(u, float(shape), float(scale), len(excess)/len(x))


def pot_var_es(fit: POTFit, alpha: float = 0.99) -> tuple[float, float]:
    """VaR=u+beta/xi*((p_u/(1-alpha))^xi-1); ES=(VaR+beta-xi*u)/(1-xi)."""
    if not 1-fit.exceedance_rate < alpha < 1 or fit.scale <= 0:
        raise ValueError("alpha must exceed threshold quantile")
    xi = fit.shape
    ratio = fit.exceedance_rate/(1-alpha)
    var = fit.threshold + (fit.scale*np.log(ratio) if abs(xi) < 1e-8 else fit.scale*np.expm1(xi*np.log(ratio))/xi)
    es = (var + fit.scale - xi*fit.threshold)/(1-xi) if xi < 1 else float("inf")
    return float(var), float(es)
