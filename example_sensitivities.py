"""Option Greek surfaces and bond portfolio parallel-shock PnL."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from quant_risk_engine.engine_sensitivities import BlackScholesEngine, YieldCurveRisk


def main() -> None:
    """Save analytic Delta/Gamma surfaces and print duration-convexity PnL."""
    spot, maturity = np.meshgrid(np.linspace(60, 140, 120), np.linspace(.02, 2, 100))
    options = BlackScholesEngine(spot, 100., maturity, .04, .25, .01)
    fig = plt.figure(figsize=(13, 5))
    for position, (name, values) in enumerate((("Delta", options.delta()), ("Gamma", options.gamma())), 1):
        ax = fig.add_subplot(1, 2, position, projection="3d")
        ax.plot_surface(spot, maturity, values, cmap="viridis", linewidth=0)
        ax.set(xlabel="Spot", ylabel="Years to expiry", zlabel=name, title=f"Call {name}")
    fig.tight_layout()
    destination = Path(__file__).resolve().parent / "outputs" / "option_greek_surfaces.png"
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)

    times = np.arange(.5, 5.5, .5)
    cashflows = np.zeros((3, len(times)))
    cashflows[0, :4] = 2.0
    cashflows[0, 3] += 100.0
    cashflows[1, :6] = 2.5
    cashflows[1, 5] += 100.0
    cashflows[2, :] = 3.0
    cashflows[2, -1] += 100.0
    rates = .035 + .003*times
    bonds = YieldCurveRisk(cashflows, times, rates)
    price = bonds.price()
    duration = bonds.modified_duration()
    convexity = bonds.effective_convexity()
    positions = np.array([1_000, 750, 500])
    shock = .01
    pnl = positions*price*(-duration*shock + .5*convexity*shock**2)
    print("Bond prices:", np.round(price, 4))
    print("Modified durations:", np.round(duration, 4))
    print("Effective convexities:", np.round(convexity, 4))
    print("PnL by bond, +100 bps:", np.round(pnl, 2))
    print("Total portfolio PnL:", round(float(pnl.sum()), 2))


if __name__ == "__main__":
    main()
