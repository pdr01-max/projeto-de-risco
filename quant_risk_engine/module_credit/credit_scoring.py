"""Altman accounting ratios and CreditMetrics Markov migration."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def _ratios(working_capital: ArrayLike, retained_earnings: ArrayLike, ebit: ArrayLike,
            equity_value: ArrayLike, total_liabilities: ArrayLike, sales: ArrayLike,
            total_assets: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    wc, re, operating, equity, debt, revenue, assets = np.broadcast_arrays(
        *[np.asarray(x, float) for x in (working_capital, retained_earnings, ebit,
                                        equity_value, total_liabilities, sales, total_assets)]
    )
    if not all(np.isfinite(x).all() for x in (wc, re, operating, equity, debt, revenue, assets)) or np.any(assets <= 0) or np.any(debt <= 0):
        raise ValueError("finite accounting inputs and positive assets/liabilities required")
    return wc/assets, re/assets, operating/assets, equity/debt, revenue/assets


def altman_z_public(working_capital: ArrayLike, retained_earnings: ArrayLike,
                    ebit: ArrayLike, market_equity: ArrayLike, total_liabilities: ArrayLike,
                    sales: ArrayLike, total_assets: ArrayLike) -> NDArray[np.float64]:
    """Public manufacturing Z=1.2X1+1.4X2+3.3X3+0.6X4+1.0X5."""
    x1, x2, x3, x4, x5 = _ratios(working_capital, retained_earnings, ebit,
                                 market_equity, total_liabilities, sales, total_assets)
    return 1.2*x1+1.4*x2+3.3*x3+.6*x4+x5


def altman_z_private(working_capital: ArrayLike, retained_earnings: ArrayLike,
                     ebit: ArrayLike, book_equity: ArrayLike, total_liabilities: ArrayLike,
                     sales: ArrayLike, total_assets: ArrayLike) -> NDArray[np.float64]:
    """Private manufacturing Z'=0.717X1+0.847X2+3.107X3+0.420X4+0.998X5."""
    x1, x2, x3, x4, x5 = _ratios(working_capital, retained_earnings, ebit,
                                 book_equity, total_liabilities, sales, total_assets)
    return .717*x1+.847*x2+3.107*x3+.420*x4+.998*x5


@dataclass(frozen=True)
class CreditMetrics:
    """Row-stochastic one-period rating transition matrix, with absorbing default."""
    transition_matrix: ArrayLike
    rating_names: tuple[str, ...]
    default_rating: str = "D"

    def _matrix(self) -> NDArray[np.float64]:
        p = np.asarray(self.transition_matrix, float)
        n = len(self.rating_names)
        if p.shape != (n, n) or len(set(self.rating_names)) != n or self.default_rating not in self.rating_names:
            raise ValueError("matrix and unique rating names must align and include default")
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-10):
            raise ValueError("transition matrix must be finite and row-stochastic")
        d = self.rating_names.index(self.default_rating)
        if not np.allclose(p[d], np.eye(n)[d], atol=1e-10):
            raise ValueError("default state must be absorbing")
        return p

    def transition_probabilities(self, periods: int) -> pd.DataFrame:
        """P(n)=P^n via matrix exponentiation, not elementwise powers."""
        if not isinstance(periods, int) or periods < 0:
            raise ValueError("periods must be a nonnegative integer")
        return pd.DataFrame(np.linalg.matrix_power(self._matrix(), periods),
                            index=self.rating_names, columns=self.rating_names)

    def cumulative_default_probability(self, periods: int) -> pd.Series:
        """Cumulative PD_i(n)=(P^n)[i,D] for absorbing default state D."""
        return self.transition_probabilities(periods)[self.default_rating].rename("cumulative_pd")
