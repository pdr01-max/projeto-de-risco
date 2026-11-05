"""Volatility estimators; forecasts are conditional on past observations only."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from arch import arch_model


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
    """Fit Student-t GARCH/GJR: h[t]=omega+(alpha+gamma I[e<0])e[t-1]^2+beta h[t-1]."""
    x = np.asarray(returns, float)
    if x.ndim != 1 or len(x) < 100 or not np.isfinite(x).all() or np.std(x) <= 0:
        raise ValueError("at least 100 nonconstant finite returns required")
    scale = 100.0
    y = x * scale
    model = arch_model(y, mean="Constant", vol="GARCH", p=1,
                       o=1 if asymmetric else 0, q=1,
                       dist="StudentsT", rescale=False)
    result = model.fit(disp="off", show_warning=False, options={"maxiter": 300})
    parameters = result.params
    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    gamma = float(parameters.get("gamma[1]", 0.))
    beta = float(parameters["beta[1]"])
    nu = float(parameters["nu"])
    mean = float(parameters["mu"])
    last_variance = float(result.conditional_volatility[-1] ** 2)
    last_residual = float(result.resid[-1])
    if not np.isfinite([omega, alpha, gamma, beta, nu, mean,
                        last_variance, last_residual]).all() or nu <= 2 or last_variance <= 0:
        raise RuntimeError("GARCH returned invalid parameters")
    return GarchFit(omega, alpha, beta, gamma, nu, mean,
                    last_variance, last_residual, scale)


def forecast_garch(fit: GarchFit) -> tuple[float, float]:
    """One-step conditional (mean, sigma) using last in-sample shock and variance."""
    h = fit.omega + (fit.alpha + fit.gamma*(fit.last_residual < 0))*fit.last_residual**2 + fit.beta*fit.last_variance
    return fit.mean/fit.scale, float(np.sqrt(max(h, 0)) / fit.scale)
