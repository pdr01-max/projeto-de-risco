"""Compound Poisson operational loss distribution approach."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray

SeverityFamily = Literal["lognormal", "pareto"]


@dataclass(frozen=True)
class LDAResult:
    annual_losses: NDArray[np.float64]
    value_at_risk: float
    expected_loss: float


@dataclass(frozen=True)
class OpRiskLDA:
    """Annual frequency N~Poisson(lambda); aggregate loss S=sum_(i=1)^N X_i."""
    annual_frequency: float
    severity_family: SeverityFamily
    severity_location: float
    severity_shape: float

    def simulate(self, simulations: int = 100_000, alpha: float = .999,
                 seed: int = 0) -> LDAResult:
        """Draw Poisson counts then severities in one vector; aggregate by scenario index.

        Lognormal X=exp(mu+sigma Z); Pareto X=x_min*(1+Y), Y~Pareto(shape).
        VaR_alpha=Q_alpha(S), with a configurable Monte Carlo sample size.
        """
        if (not np.isfinite([self.annual_frequency, self.severity_location,
                             self.severity_shape]).all() or self.annual_frequency < 0
                or self.severity_shape <= 0 or self.severity_family not in ("lognormal", "pareto")
                or (self.severity_family == "pareto" and self.severity_location <= 0)
                or not isinstance(simulations, int) or simulations < 1000
                or not 0 < alpha < 1):
            raise ValueError("invalid LDA inputs")
        rng = np.random.default_rng(seed)
        counts = rng.poisson(self.annual_frequency, size=simulations)
        scenario = np.repeat(np.arange(simulations), counts)
        if self.severity_family == "lognormal":
            severity = rng.lognormal(self.severity_location, self.severity_shape, size=len(scenario))
        else:
            severity = self.severity_location*(1+rng.pareto(self.severity_shape, size=len(scenario)))
        aggregate = np.bincount(scenario, weights=severity, minlength=simulations)
        return LDAResult(aggregate, float(np.quantile(aggregate, alpha)), float(aggregate.mean()))
