"""Synthetic Merton and 1,000-loan corporate IRB illustration."""
import numpy as np
from quant_risk_engine.module_credit import MertonModel, VasicekIRB


def main() -> None:
    """Calibrate a leveraged issuer and calculate vectorized IRB portfolio capital."""
    issuer = MertonModel(equity_value=30., equity_volatility=.75,
                         debt_face_value=100., maturity=1., risk_free_rate=.04,
                         asset_drift=.03).solve()
    print(f"Merton assets={issuer.asset_value:.3f}, asset vol={issuer.asset_volatility:.3%}, "
          f"DD={issuer.distance_to_default:.3f}, PD={issuer.probability_of_default:.3%}")

    rng = np.random.default_rng(2026)
    loans = 1_000
    pd = np.clip(rng.lognormal(mean=np.log(.025), sigma=.8, size=loans), .0005, .4)
    lgd = rng.beta(4, 5, size=loans)
    ead = rng.lognormal(mean=np.log(750_000), sigma=.65, size=loans)
    maturity = rng.uniform(1., 5., size=loans)
    sales = rng.lognormal(mean=np.log(35), sigma=.8, size=loans)
    result = VasicekIRB(pd, lgd, ead, maturity, sales).calculate()
    for name, amount in result.totals.items():
        print(f"{name}: {amount:,.2f}")


if __name__ == "__main__":
    main()
