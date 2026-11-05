"""Reproducible synthetic validation: normal versus GARCH+POT+Student-t copula."""
import numpy as np
from scipy.stats import t
from quant_risk_engine.core.estimators import estimate_moments
from quant_risk_engine.engine_market_tail import (
    backtest_var, fit_copula, fit_garch, fit_pot, forecast_garch,
    normal_var_es, pot_var_es, simulate_copula,
)


def synthetic_returns(n: int = 950, seed: int = 42) -> np.ndarray:
    """Correlated Student-t innovations with persistent GJR volatility clustering."""
    rng = np.random.default_rng(seed)
    corr = np.array([[1., .55], [.55, 1.]])
    z = rng.multivariate_normal(np.zeros(2), corr, size=n)
    z /= np.sqrt(rng.chisquare(5, size=(n, 1))/5)
    z *= np.sqrt(3/5)
    h = np.full((n+1, 2), .0001)
    r = np.empty((n, 2))
    for i in range(n):
        r[i] = np.sqrt(h[i])*z[i]
        h[i+1] = .000003 + .045*r[i]**2 + .045*(r[i]<0)*r[i]**2 + .88*h[i]
    return r


def compare(window: int = 500, step: int = 10, alpha: float = .975) -> None:
    """Walk forward; refit every step on the trailing window and forecast subsequent dates."""
    returns = synthetic_returns()
    weights = np.array([.5, .5])
    normal_forecasts, hybrid_forecasts, realized = [], [], []
    first = True
    for start in range(window, len(returns), step):
        train = returns[start-window:start]
        horizon = min(step, len(returns)-start)
        mu, cov = estimate_moments(train)
        normal_var, normal_es = normal_var_es(float(mu@weights), float(np.sqrt(weights@cov@weights)), alpha)
        fits = [fit_garch(train[:, j]) for j in range(2)]
        forecasts = [forecast_garch(f) for f in fits]
        means = np.array([f[0] for f in forecasts])
        sigmas = np.array([f[1] for f in forecasts])
        standardized = (train-means)/sigmas
        copula = fit_copula(standardized, "student_t", 6)
        u = simulate_copula(copula, 20_000, seed=start)
        scenarios = t.ppf(u, df=6)*np.sqrt(4/6)*sigmas + means
        simulated_loss = -(scenarios@weights)
        pot = fit_pot(simulated_loss, .90)
        hybrid_var, hybrid_es = pot_var_es(pot, alpha)
        if first:
            print(f"VaR/ES Normal: {normal_var:.5f} / {normal_es:.5f}")
            print(f"VaR/ES GARCH+EVT+Student-t Copula: {hybrid_var:.5f} / {hybrid_es:.5f}")
            first = False
        normal_forecasts.extend([normal_var]*horizon)
        hybrid_forecasts.extend([hybrid_var]*horizon)
        realized.extend(returns[start:start+horizon]@weights)
    for name, forecast in [("Normal", normal_forecasts), ("GARCH+EVT+Student-t Copula", hybrid_forecasts)]:
        kupiec, independence = backtest_var(realized, forecast, alpha)
        print(f"{name}: breaches={kupiec.breaches}/{kupiec.observations}, Kupiec LR={kupiec.statistic:.3f}, p={kupiec.p_value:.3f}; Christoffersen p={independence.p_value:.3f}")


if __name__ == "__main__":
    compare()
