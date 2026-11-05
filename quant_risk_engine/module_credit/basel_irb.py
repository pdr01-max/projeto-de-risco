"""Basel corporate foundation/advanced IRB single-factor capital formula."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class IRBResult:
    expected_loss: NDArray[np.float64]
    unexpected_loss_capital: NDArray[np.float64]
    risk_weighted_assets: NDArray[np.float64]
    asset_correlation: NDArray[np.float64]

    @property
    def totals(self) -> dict[str, float]:
        """Portfolio sums; EAD and outputs must share one currency unit."""
        return {"expected_loss": float(self.expected_loss.sum()),
                "unexpected_loss_capital": float(self.unexpected_loss_capital.sum()),
                "risk_weighted_assets": float(self.risk_weighted_assets.sum())}


@dataclass(frozen=True)
class VasicekIRB:
    """Corporate IRB capital at 99.9% confidence; no regulatory output floor applied."""
    pd: ArrayLike
    lgd: ArrayLike
    ead: ArrayLike
    maturity: ArrayLike = 2.5
    annual_sales_millions: ArrayLike | None = None

    def calculate(self) -> IRBResult:
        """EL=PD*LGD*EAD; K=[LGD*N((N^-1(PD)+sqrt(R)N^-1(.999))/sqrt(1-R))-PD*LGD]*MA.

        R=.12*f(PD)+.24*(1-f(PD))-SME_adj, f=(1-exp(-50PD))/(1-exp(-50));
        SME_adj=.04*(1-(S-5)/45), S clipped to [5,50] million.
        MA=[1+(M-2.5)b]/[1-1.5b], b=(.11852-.05478 ln(PD))²;
        RWA=12.5*K*EAD. PD floor of 0.05% is applied to non-defaulted
        corporate exposures; PD=1 defaulted exposures require separate treatment.
        """
        pd, lgd, ead, maturity = np.broadcast_arrays(
            *[np.asarray(x, dtype=float) for x in (self.pd, self.lgd, self.ead, self.maturity)]
        )
        if not all(np.isfinite(x).all() for x in (pd, lgd, ead, maturity)):
            raise ValueError("IRB inputs must be finite")
        if np.any((pd <= 0) | (pd >= 1)) or np.any((lgd < 0) | (lgd > 1)) or np.any(ead < 0) or np.any((maturity < 1) | (maturity > 5)):
            raise ValueError("require 0<PD<1, 0<=LGD<=1, EAD>=0 and 1<=M<=5")
        effective_pd = np.maximum(pd, .0005)
        factor = -np.expm1(-50*effective_pd)/(-np.expm1(-50.))
        correlation = .12*factor + .24*(1-factor)
        if self.annual_sales_millions is not None:
            sales = np.broadcast_to(np.asarray(self.annual_sales_millions, float), pd.shape)
            if not np.isfinite(sales).all() or np.any(sales <= 0):
                raise ValueError("sales must be positive and finite")
            correlation -= .04*(1-(np.clip(sales, 5, 50)-5)/45)
        b = (.11852-.05478*np.log(effective_pd))**2
        maturity_adjustment = (1+(maturity-2.5)*b)/(1-1.5*b)
        conditional_pd = norm.cdf((norm.ppf(effective_pd)+np.sqrt(correlation)*norm.ppf(.999))/np.sqrt(1-correlation))
        capital_rate = np.maximum(0, lgd*(conditional_pd-effective_pd)*maturity_adjustment)
        return IRBResult(pd*lgd*ead, capital_rate*ead, 12.5*capital_rate*ead, correlation)
