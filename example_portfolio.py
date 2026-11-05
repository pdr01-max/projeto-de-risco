"""Compare five-asset Markowitz, equal risk contribution and HRP allocations."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from quant_risk_engine.core.estimators import ledoit_wolf_covariance
from quant_risk_engine.engine_portfolio import MeanVarianceOptimizer, RiskParityOptimizer, HRPOptimizer


def main() -> None:
    """Simulate correlated daily returns and save a comparison chart."""
    rng = np.random.default_rng(47)
    names = ["Equities", "Bonds", "Credit", "Commodities", "REITs"]
    vol = np.array([.018, .005, .009, .014, .016])
    factors = rng.standard_normal((750, 3))
    loading = np.array([[.9,.1,.1],[-.2,.5,.1],[.5,.5,.1],[.1,-.1,.8],[.7,.1,.3]])
    returns = pd.DataFrame((factors@loading.T + rng.standard_normal((750,5))*.45)*vol + np.array([.0005,.00015,.0003,.00035,.0004]), columns=names)
    covariance = ledoit_wolf_covariance(returns.to_numpy())
    allocations = pd.DataFrame({
        "Markowitz": MeanVarianceOptimizer(risk_aversion=10).optimize(returns, covariance=covariance),
        "Risk Parity": RiskParityOptimizer().optimize(returns, covariance=covariance),
        "HRP": HRPOptimizer().optimize(returns, covariance=covariance),
    })
    print(allocations.round(4))
    ax = allocations.plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylabel("Portfolio weight")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.25)
    ax.figure.tight_layout()
    destination = Path(__file__).resolve().parent / "outputs" / "portfolio_allocations.png"
    destination.parent.mkdir(exist_ok=True)
    ax.figure.savefig(destination, dpi=160)
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
