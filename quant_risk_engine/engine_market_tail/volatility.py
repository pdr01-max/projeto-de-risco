"""Volatility estimators; forecasts are conditional on past observations only."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import t


def ewma_volatility(returns: ArrayLike, decay: float = 0.94, initial_variance: float | None = None) -> NDArray[np.float64]:
    """One-step forecasts: sigma[t]^2=lambda*sigma[t-1]^2+(1-lambda)*r[t-1]^2."""
    x = np.asarray(returns, float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or not 0 < decay < 1:
        raise ValueError("invalid EWMA inputs")
    v = float(np.var(x[:min(30, len(x))], ddof=1)) if initial_variance is None else initial_variance
    if v <= 0:
        raise ValueError("initial variance must be positive")
    out = np.empty(len(x) + 1)
    out[0] = v
    for i in range(len(x)):
        out[i+1] = decay*out[i] + (1-decay)*x[i]**2
    return np.sqrt(out[1:])


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    gamma: float
    nu: float
    mean: float
    last_variance: float
    last_residual: float
    scale: float


def fit_garch(returns: ArrayLike, *, asymmetric: bool = False) -> GarchFit:
    """Fit Student-t GARCH/GJR: h[t]=omega+alpha*e[t-1]^2+gamma*I(e<0)*e^2+beta*h[t-1]."""
    x = np.asarray(returns, float)
    if x.ndim != 1 or len(x) < 100 or not np.isfinite(x).all() or np.std(x) <= 0:
        raise ValueError("at least 100 nonconstant finite returns required")
    scale = 100.0
    y = x*scale
    sample_variance = float(np.var(y, ddof=1))

    def likelihood(p: NDArray[np.float64]) -> float:
        omega, alpha, gamma, beta, nu, mean = p
        if alpha + beta + gamma/2 >= .999:
            return 1e12
        e = y-mean
        h = np.empty(len(y))
        h[0] = sample_variance
        for i in range(1, len(y)):
            h[i] = omega + (alpha + gamma*(e[i-1] < 0))*e[i-1]**2 + beta*h[i-1]
        standardized = e/np.sqrt(h)
        return float(np.sum(.5*np.log(h)-t.logpdf(standardized*np.sqrt(nu/(nu-2)), nu)+.5*np.log((nu-2)/nu)))

    initial = np.array([sample_variance*.05, .06, .04 if asymmetric else 0., .87, 7., float(y.mean())])
    bounds = [(1e-9, sample_variance*10), (1e-6, .5), (0., .5) if asymmetric else (0., 0.), (.01, .999), (2.05, 100.), (float(y.mean()-5*np.std(y)), float(y.mean()+5*np.std(y)))]
    result = minimize(likelihood, initial, method="L-BFGS-B", bounds=bounds)
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError("GARCH optimization failed")
    omega, alpha, gamma, beta, nu, mean = result.x
    e = y-mean
    h_last = sample_variance
    for shock in e[:-1]:
        h_last = omega + (alpha + gamma*(shock < 0))*shock**2 + beta*h_last
    return GarchFit(float(omega), float(alpha), float(beta), float(gamma), float(nu), float(mean), float(h_last), float(e[-1]), scale)


def forecast_garch(fit: GarchFit) -> tuple[float, float]:
    """One-step conditional (mean, sigma) using last in-sample shock and variance."""
    h = fit.omega + (fit.alpha + fit.gamma*(fit.last_residual < 0))*fit.last_residual**2 + fit.beta*fit.last_variance
    return fit.mean/fit.scale, float(np.sqrt(max(h, 0)) / fit.scale)
