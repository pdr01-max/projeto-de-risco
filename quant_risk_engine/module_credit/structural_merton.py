"""Merton equity inversion and theoretical default probability."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root
from scipy.stats import norm


@dataclass(frozen=True)
class MertonResult:
    asset_value: float
    asset_volatility: float
    distance_to_default: float
    probability_of_default: float
    equity_residual: float
    volatility_residual: float


@dataclass(frozen=True)
class MertonModel:
    """Equity is a call on assets struck at debt face value D, maturing at T."""
    equity_value: float
    equity_volatility: float
    debt_face_value: float
    maturity: float
    risk_free_rate: float
    asset_drift: float | None = None

    def solve(self) -> MertonResult:
        """Solve E=V N(d1)-D exp(-rT)N(d2), sigma_E=V N(d1)sigma_V/E.

        DD=[ln(V/D)+(mu_V-sigma_V²/2)T]/(sigma_V sqrt(T)); PD=N(-DD).
        If physical asset drift is unavailable, use r as the theoretical drift.
        """
        e, se, d, t, r = (self.equity_value, self.equity_volatility,
                          self.debt_face_value, self.maturity, self.risk_free_rate)
        if not np.isfinite([e, se, d, t, r]).all() or min(e, se, d, t) <= 0:
            raise ValueError("positive finite equity, volatility, debt and maturity required")
        drift = r if self.asset_drift is None else self.asset_drift
        if not np.isfinite(drift):
            raise ValueError("asset drift must be finite")

        def equations(log_parameters: np.ndarray) -> np.ndarray:
            v, sv = np.exp(log_parameters)
            d1 = (np.log(v/d)+(r+.5*sv**2)*t)/(sv*np.sqrt(t))
            d2 = d1-sv*np.sqrt(t)
            model_equity = v*norm.cdf(d1)-d*np.exp(-r*t)*norm.cdf(d2)
            model_equity_vol = v*norm.cdf(d1)*sv/e
            return np.array([(model_equity-e)/e, (model_equity_vol-se)/se])

        guess = np.log([e+d*np.exp(-r*t), max(.01, se*e/(e+d))])
        solution = root(equations, guess)
        if not solution.success or not np.isfinite(solution.x).all():
            raise RuntimeError(f"Merton calibration failed: {solution.message}")
        residual = equations(solution.x)
        if np.max(np.abs(residual)) > 1e-7:
            raise RuntimeError("Merton calibration residual exceeds tolerance")
        v, sv = np.exp(solution.x)
        dd = (np.log(v/d)+(drift-.5*sv**2)*t)/(sv*np.sqrt(t))
        return MertonResult(float(v), float(sv), float(dd), float(norm.cdf(-dd)),
                            float(residual[0]), float(residual[1]))
