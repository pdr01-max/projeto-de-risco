"""Vectorized Black-Scholes-Merton pricing and analytical sensitivities."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesEngine:
    """European options under continuous risk-free rate r and dividend yield q."""
    spot: ArrayLike
    strike: ArrayLike
    maturity: ArrayLike
    rate: ArrayLike
    volatility: ArrayLike
    dividend_yield: ArrayLike = 0.0

    def _terms(self) -> tuple[NDArray[np.float64], ...]:
        s, k, t, r, vol, q = np.broadcast_arrays(
            *[np.asarray(v, dtype=float) for v in (
                self.spot, self.strike, self.maturity, self.rate,
                self.volatility, self.dividend_yield,
            )]
        )
        if not all(np.isfinite(v).all() for v in (s, k, t, r, vol, q)):
            raise ValueError("all option inputs must be finite")
        if np.any(s <= 0) or np.any(k <= 0) or np.any(t <= 0) or np.any(vol <= 0):
            raise ValueError("spot, strike, maturity and volatility must be positive")
        root_t = np.sqrt(t)
        d1 = (np.log(s / k) + (r - q + .5 * vol**2) * t) / (vol * root_t)
        d2 = d1 - vol * root_t
        return s, k, t, r, vol, q, d1, d2

    @staticmethod
    def _side(option_type: OptionType) -> int:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        return 1 if option_type == "call" else -1

    def price(self, option_type: OptionType = "call") -> NDArray[np.float64]:
        """BSM: V=phi[S exp(-qT) N(phi d1)-K exp(-rT) N(phi d2)]."""
        s, k, t, r, _, q, d1, d2 = self._terms()
        side = self._side(option_type)
        return side * (s * np.exp(-q*t) * norm.cdf(side*d1) - k * np.exp(-r*t) * norm.cdf(side*d2))

    def delta(self, option_type: OptionType = "call") -> NDArray[np.float64]:
        """Delta=dV/dS=phi exp(-qT) N(phi d1)."""
        _, _, t, _, _, q, d1, _ = self._terms()
        side = self._side(option_type)
        return side * np.exp(-q*t) * norm.cdf(side*d1)

    def gamma(self) -> NDArray[np.float64]:
        """Gamma=d²V/dS²=exp(-qT) phi_normal(d1)/(S sigma sqrt(T))."""
        s, _, t, _, vol, q, d1, _ = self._terms()
        return np.exp(-q*t) * norm.pdf(d1) / (s * vol * np.sqrt(t))

    def vega(self) -> NDArray[np.float64]:
        """Vega=dV/dsigma=S exp(-qT) phi_normal(d1) sqrt(T), per 1.0 vol."""
        s, _, t, _, _, q, d1, _ = self._terms()
        return s * np.exp(-q*t) * norm.pdf(d1) * np.sqrt(t)

    def theta(self, option_type: OptionType = "call") -> NDArray[np.float64]:
        """Calendar theta=dV/dt with remaining maturity T decreasing as time passes."""
        s, k, t, r, vol, q, d1, d2 = self._terms()
        side = self._side(option_type)
        return (-s*np.exp(-q*t)*norm.pdf(d1)*vol/(2*np.sqrt(t))
                - side*r*k*np.exp(-r*t)*norm.cdf(side*d2)
                + side*q*s*np.exp(-q*t)*norm.cdf(side*d1))

    def rho(self, option_type: OptionType = "call") -> NDArray[np.float64]:
        """Rho=dV/dr=phi K T exp(-rT) N(phi d2), per 1.0 rate."""
        _, k, t, r, _, _, _, d2 = self._terms()
        side = self._side(option_type)
        return side * k * t * np.exp(-r*t) * norm.cdf(side*d2)

    def vanna(self) -> NDArray[np.float64]:
        """Vanna=d²V/(dS d sigma)=-exp(-qT) phi_normal(d1) d2/sigma."""
        _, _, t, _, vol, q, d1, d2 = self._terms()
        return -np.exp(-q*t) * norm.pdf(d1) * d2 / vol
