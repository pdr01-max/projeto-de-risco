"""Risk evaluation on fitted parameters or scenarios; positive loss convention."""
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from quant_risk_engine.core.matrix import repair_psd


def _alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1:
        raise ValueError("confidence alpha must lie in (0.5,1)")


def normal_var_es(mean: float, volatility: float, alpha: float = 0.99) -> tuple[float, float]:
    """For L=-R normal: VaR=-mu+sigma*z_alpha; ES=-mu+sigma*phi(z_alpha)/(1-alpha)."""
    _alpha(alpha)
    if volatility < 0 or not np.isfinite([mean, volatility]).all():
        raise ValueError("invalid normal parameters")
    z = norm.ppf(alpha)
    return float(-mean + volatility*z), float(-mean + volatility*norm.pdf(z)/(1-alpha))


def historical_var_es(returns: ArrayLike, alpha: float = 0.99) -> tuple[float, float]:
    """Empirical VaR=Q_alpha(-R), ES=mean(L | L>=VaR)."""
    _alpha(alpha)
    loss = -np.asarray(returns, dtype=float)
    if loss.ndim != 1 or len(loss) < 20 or not np.isfinite(loss).all():
        raise ValueError("at least 20 finite returns required")
    var = float(np.quantile(loss, alpha))
    return var, float(loss[loss >= var].mean())


def monte_carlo_var_es(mean: ArrayLike, covariance: ArrayLike, weights: ArrayLike, *, alpha: float = 0.99, n: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """Draw R~N(mu,Sigma), evaluate portfolio loss L=-R w and empirical VaR/ES."""
    _alpha(alpha)
    mu, w = np.asarray(mean, float), np.asarray(weights, float)
    cov = repair_psd(covariance)
    if mu.ndim != 1 or w.shape != mu.shape or cov.shape != (len(mu), len(mu)) or n < 1000:
        raise ValueError("incompatible dimensions or insufficient draws")
    draws = np.random.default_rng(seed).multivariate_normal(mu, cov, size=n)
    return historical_var_es(draws @ w, alpha)


def liquidity_var(var: float, spread: float, position: float, liquidation_days: float = 1) -> float:
    """LVaR=VaR*sqrt(h)+|position|*spread/2; VaR and position in same currency units."""
    if min(var, spread, liquidation_days) < 0 or not np.isfinite([var, spread, position, liquidation_days]).all():
        raise ValueError("invalid liquidity inputs")
    return float(var*np.sqrt(liquidation_days) + abs(position)*spread/2)
