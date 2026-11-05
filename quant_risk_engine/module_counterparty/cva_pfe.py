"""Uncollateralized counterparty exposure and unilateral CVA."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CounterpartyResult:
    expected_exposure: NDArray[np.float64]
    potential_future_exposure: NDArray[np.float64]
    cva: float


@dataclass(frozen=True)
class CounterpartyRisk:
    """Rows are Monte Carlo MtM paths; columns are increasing future dates."""
    mtm_paths: ArrayLike
    times: ArrayLike
    cumulative_default_probability: ArrayLike
    recovery_rate: float = .4
    discount_factors: ArrayLike | None = None

    def calculate(self, pfe_quantile: float = .95) -> CounterpartyResult:
        """EE_t=mean(max(V_t,0)); PFE_t=Q_q(max(V_t,0)); CVA=(1-R) sum_t DF_t EE_t ΔPD_t.

        Defaults are assumed independent of exposure. No collateral, netting,
        wrong-way risk or discounting is included unless supplied explicitly.
        """
        paths = np.asarray(self.mtm_paths, float)
        times = np.asarray(self.times, float)
        cumulative_pd = np.asarray(self.cumulative_default_probability, float)
        if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 1 or not np.isfinite(paths).all():
            raise ValueError("finite Monte Carlo matrix (paths x dates) required")
        n = paths.shape[1]
        if times.shape != (n,) or cumulative_pd.shape != (n,) or not np.isfinite(times).all() or not np.isfinite(cumulative_pd).all():
            raise ValueError("times and cumulative PD must match simulation dates")
        if np.any(times <= 0) or np.any(np.diff(times) <= 0) or np.any(np.diff(cumulative_pd) < -1e-12) or np.any((cumulative_pd < 0) | (cumulative_pd > 1)):
            raise ValueError("times must increase; cumulative PD must increase within [0,1]")
        if not 0 <= self.recovery_rate <= 1 or not 0 < pfe_quantile < 1:
            raise ValueError("invalid recovery or PFE quantile")
        discounts = np.ones(n) if self.discount_factors is None else np.asarray(self.discount_factors, float)
        if discounts.shape != (n,) or not np.isfinite(discounts).all() or np.any(discounts <= 0):
            raise ValueError("positive finite discount factors required")
        exposure = np.maximum(paths, 0)
        ee = exposure.mean(axis=0)
        pfe = np.quantile(exposure, pfe_quantile, axis=0)
        incremental_pd = np.diff(np.r_[0., cumulative_pd])
        cva = (1-self.recovery_rate)*np.sum(discounts*ee*incremental_pd)
        return CounterpartyResult(ee, pfe, float(cva))
