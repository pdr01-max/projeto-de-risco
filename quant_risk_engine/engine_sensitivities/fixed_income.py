"""Vectorized cash-flow and term-structure sensitivities."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class YieldCurveRisk:
    """Nominal annual zero rates compounded frequency times per year; rows are bonds."""
    cashflows: ArrayLike
    times: ArrayLike
    zero_rates: ArrayLike
    frequency: int = 2

    def _inputs(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        cf = np.asarray(self.cashflows, dtype=float)
        if cf.ndim == 1:
            cf = cf[None, :]
        if cf.ndim != 2 or not np.isfinite(cf).all() or np.any(cf < 0) or np.any(cf.sum(axis=1) <= 0):
            raise ValueError("cashflows must be nonnegative N x M with positive row sums")
        try:
            times = np.broadcast_to(np.asarray(self.times, dtype=float), cf.shape)
            rates = np.broadcast_to(np.asarray(self.zero_rates, dtype=float), cf.shape)
        except ValueError as exc:
            raise ValueError("times and zero_rates must broadcast to cashflows") from exc
        if not np.isfinite(times).all() or not np.isfinite(rates).all() or np.any(times < 0):
            raise ValueError("finite nonnegative payment times and finite rates required")
        if not isinstance(self.frequency, int) or self.frequency < 1 or np.any(1 + rates/self.frequency <= 0):
            raise ValueError("invalid compounding frequency or rates")
        return cf, times, rates

    def _present_values(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        cf, times, rates = self._inputs()
        pv = cf * (1 + rates/self.frequency)**(-self.frequency*times)
        return pv, times, rates

    def price(self) -> NDArray[np.float64]:
        """P=sum_j CF_j (1+y_j/m)^(-m t_j)."""
        pv, _, _ = self._present_values()
        return pv.sum(axis=1)

    def macaulay_duration(self) -> NDArray[np.float64]:
        """D_Mac=sum_j t_j PV_j/P, in years."""
        pv, times, _ = self._present_values()
        return np.sum(times*pv, axis=1)/pv.sum(axis=1)

    def modified_duration(self) -> NDArray[np.float64]:
        """D_Mod=-P^-1 dP/ds=sum_j [t_j/(1+y_j/m)] PV_j/P for parallel zero-rate shift s."""
        pv, times, rates = self._present_values()
        return np.sum(times/(1+rates/self.frequency)*pv, axis=1)/pv.sum(axis=1)

    def effective_convexity(self) -> NDArray[np.float64]:
        """C=P^-1 d²P/ds²=sum_j t_j(t_j+1/m) PV_j/[(1+y_j/m)² P]."""
        pv, times, rates = self._present_values()
        factor = times*(times+1/self.frequency)/(1+rates/self.frequency)**2
        return np.sum(factor*pv, axis=1)/pv.sum(axis=1)

    def pv01(self) -> NDArray[np.float64]:
        """PV01=-dP/ds*10^-4=P*D_Mod*10^-4, positive for long bonds."""
        return self.price()*self.modified_duration()*1e-4

    def key_rate_duration(self, vertices: ArrayLike) -> NDArray[np.float64]:
        """KRD_k=-P^-1 dP/dz_k using linear hat functions on curve vertices.

        Returns N x K. Vertex shocks interpolate linearly between neighboring
        tenors and extend constantly beyond the first and last vertex.
        """
        pv, times, rates = self._present_values()
        knots = np.asarray(vertices, dtype=float)
        if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0):
            raise ValueError("at least two strictly increasing finite vertices required")
        right = np.clip(np.searchsorted(knots, times, side="right"), 1, len(knots)-1)
        left = right-1
        upper_weight = np.clip((times-knots[left])/(knots[right]-knots[left]), 0, 1)
        lower_weight = 1-upper_weight
        basis = np.eye(len(knots))
        hats = lower_weight[..., None]*basis[left] + upper_weight[..., None]*basis[right]
        sensitivity = pv*times/(1+rates/self.frequency)
        return np.sum(sensitivity[..., None]*hats, axis=1)/pv.sum(axis=1)[:, None]
