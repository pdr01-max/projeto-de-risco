"""Discrete Almgren-Chriss liquidation schedule."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OptimalExecution:
    """Liquidate X shares in N periods over T; temporary impact eta and permanent gamma."""
    order_size: float
    periods: int
    horizon: float
    risk_aversion: float
    volatility: float
    temporary_impact: float
    permanent_impact: float = 0.0

    def schedule(self) -> NDArray[np.float64]:
        """x_j=X*sinh[kappa(T-t_j)]/sinh(kappa T); n_j=x_(j-1)-x_j.

        kappa=arccosh(1+lambda*sigma²*dt²/(2*(eta-gamma*dt/2)))/dt.
        In the zero-risk limit n_j=X/N. Shares sold are nonnegative and sum to X.
        """
        x, n, horizon = self.order_size, self.periods, self.horizon
        lam, sigma, eta, gamma = (self.risk_aversion, self.volatility,
                                   self.temporary_impact, self.permanent_impact)
        if not np.isfinite([x, horizon, lam, sigma, eta, gamma]).all() or x <= 0 or horizon <= 0 or lam < 0 or sigma < 0 or eta <= 0 or gamma < 0 or not isinstance(n, int) or n < 1:
            raise ValueError("invalid Almgren-Chriss parameters")
        dt = horizon/n
        adjusted_eta = eta-gamma*dt/2
        if adjusted_eta <= 0:
            raise ValueError("temporary impact must exceed gamma*dt/2")
        if lam == 0 or sigma == 0:
            return np.full(n, x/n)
        kappa_dt = np.arccosh(1+lam*sigma**2*dt**2/(2*adjusted_eta))
        grid = np.arange(n+1, dtype=float)
        # Stable sinh ratio for large risk aversion, avoiding overflow.
        a = kappa_dt*(n-grid)
        b = kappa_dt*n
        holdings = x*np.exp(a-b)*(-np.expm1(-2*a))/(-np.expm1(-2*b))
        holdings[-1] = 0.
        sales = -np.diff(holdings)
        return sales*x/sales.sum()
