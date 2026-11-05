"""Systemic tail dependence using MES and linear quantile-regression CoVaR."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, csc_matrix


@dataclass(frozen=True)
class CoVaRResult:
    covar_distress: float
    covar_median: float
    delta_covar: float
    quantile_slope: float


@dataclass(frozen=True)
class SystemicRisk:
    """Aligned institution and system returns; negative returns represent losses."""
    institution_returns: ArrayLike
    system_returns: ArrayLike

    def _inputs(self) -> tuple[np.ndarray, np.ndarray]:
        institution = np.asarray(self.institution_returns, float)
        system = np.asarray(self.system_returns, float)
        if institution.ndim != 1 or system.shape != institution.shape or len(system) < 30 or not np.isfinite(institution).all() or not np.isfinite(system).all():
            raise ValueError("at least 30 aligned finite return observations required")
        return institution, system

    def marginal_expected_shortfall(self, tail_probability: float = .05) -> float:
        """MES_i=-E[R_i | R_system <= Q_alpha(R_system)], positive loss convention."""
        institution, system = self._inputs()
        if not 0 < tail_probability < .5:
            raise ValueError("tail_probability must lie in (0,0.5)")
        cutoff = np.quantile(system, tail_probability)
        return float(-institution[system <= cutoff].mean())

    def delta_covar(self, tail_probability: float = .05) -> CoVaRResult:
        """Q_alpha(R_sys | R_i=x)=a+b*x via pinball-loss linear programming.

        CoVaR_distress=-(a+b*Q_alpha(R_i)); CoVaR_median=-(a+b*median(R_i));
        DeltaCoVaR=CoVaR_distress-CoVaR_median, in positive-loss units.
        """
        institution, system = self._inputs()
        if not 0 < tail_probability < .5:
            raise ValueError("tail_probability must lie in (0,0.5)")
        n = len(system)
        design = csc_matrix(np.column_stack([np.ones(n), institution]))
        constraints = hstack([design, eye(n, format="csc"), -eye(n, format="csc")], format="csc")
        objective = np.r_[np.zeros(2), np.full(n, tail_probability), np.full(n, 1-tail_probability)]
        result = linprog(objective, A_eq=constraints, b_eq=system,
                         bounds=[(None, None)]*2+[(0, None)]*(2*n), method="highs")
        if not result.success:
            raise RuntimeError(f"quantile regression failed: {result.message}")
        intercept, slope = result.x[:2]
        distress = -(intercept+slope*np.quantile(institution, tail_probability))
        median = -(intercept+slope*np.median(institution))
        return CoVaRResult(float(distress), float(median), float(distress-median), float(slope))
