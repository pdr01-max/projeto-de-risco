"""Black-Litterman posterior estimation and fractional Kelly allocation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from quant_risk_engine.core.matrix import repair_psd
from .optimizers_convex import MeanVarianceOptimizer


@dataclass(frozen=True)
class BlackLitterman:
    """Posterior mu=[(tau Sigma)^-1+P'Omega^-1 P]^-1[(tau Sigma)^-1 pi+P'Omega^-1 q]."""
    risk_aversion: float = 2.5
    tau: float = .05

    def posterior(self, covariance: ArrayLike, market_weights: ArrayLike, views: ArrayLike, view_returns: ArrayLike, view_covariance: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        sigma = repair_psd(covariance)
        market = np.asarray(market_weights, float)
        p, q = np.asarray(views, float), np.asarray(view_returns, float)
        n = len(sigma)
        if market.shape != (n,) or p.ndim != 2 or p.shape[1] != n or q.shape != (len(p),) or not np.isfinite(np.r_[market, p.ravel(), q]).all() or self.tau <= 0 or self.risk_aversion <= 0:
            raise ValueError("invalid Black-Litterman inputs")
        prior = self.risk_aversion*sigma@market
        tau_sigma = self.tau*sigma
        omega = repair_psd(view_covariance if view_covariance is not None else np.diag(np.diag(p@tau_sigma@p.T)))
        middle = repair_psd(p@tau_sigma@p.T+omega)
        gain = np.linalg.solve(middle, p@tau_sigma).T
        posterior_mean = prior + gain@(q-p@prior)
        posterior_covariance = repair_psd(sigma + tau_sigma - gain@p@tau_sigma)
        return posterior_mean, posterior_covariance

    def optimize(self, covariance: ArrayLike, market_weights: ArrayLike, views: ArrayLike, view_returns: ArrayLike, *, view_covariance: ArrayLike | None = None, asset_names: list[str] | None = None, max_weight: float = 1.) -> pd.Series:
        mu, sigma = self.posterior(covariance, market_weights, views, view_returns, view_covariance)
        result = MeanVarianceOptimizer(self.risk_aversion, max_weight).optimize(mean=mu, covariance=sigma)
        if asset_names is not None:
            if len(asset_names) != len(result):
                raise ValueError("asset_names length mismatch")
            result.index = asset_names
        return result


@dataclass(frozen=True)
class FractionalKelly:
    """Fractional risky exposure f=fraction*Sigma^-1 mu; residual wealth stays in cash."""
    fraction: float = .5

    def optimize(self, mean: ArrayLike, covariance: ArrayLike, *, asset_names: list[str] | None = None) -> pd.Series:
        mu = np.asarray(mean, float)
        sigma = repair_psd(covariance)
        if mu.ndim != 1 or sigma.shape != (len(mu), len(mu)) or not np.isfinite(mu).all() or not 0 < self.fraction <= 1:
            raise ValueError("invalid Kelly inputs")
        raw = self.fraction*np.linalg.solve(sigma, mu)
        positive = np.maximum(raw, 0)
        names = asset_names if asset_names is not None else [f"asset_{i}" for i in range(len(mu))]
        if len(names) != len(mu):
            raise ValueError("asset_names length mismatch")
        if "cash" in names:
            raise ValueError("cash is a reserved asset name")
        risky = positive/max(1., positive.sum())
        return pd.Series(np.r_[risky, 1-risky.sum()], index=[*names, "cash"], name="weight")
