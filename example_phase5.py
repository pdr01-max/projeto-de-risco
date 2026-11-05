"""Counterparty PFE, Almgren-Chriss schedules and operational LDA demonstration."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from quant_risk_engine.module_counterparty import CounterpartyRisk
from quant_risk_engine.module_execution import OptimalExecution
from quant_risk_engine.module_systemic_ops import OpRiskLDA


def main() -> None:
    """Simulate derivative MtM paths; save PFE and execution figures; print LDA VaR."""
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(17)
    times = np.linspace(.05, 2., 40)
    shocks = rng.standard_normal((20_000, len(times)))
    underlying = 100*np.exp(np.cumsum((.02-.5*.25**2)*np.diff(np.r_[0, times]) + .25*np.sqrt(np.diff(np.r_[0, times]))*shocks, axis=1))
    mtm = (underlying-100)*10_000
    cumulative_pd = 1-np.exp(-.025*times)
    exposure = CounterpartyRisk(mtm, times, cumulative_pd, .4, np.exp(-.04*times)).calculate()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, exposure.potential_future_exposure, label="PFE 95%")
    ax.plot(times, exposure.expected_exposure, label="EE")
    ax.set(xlabel="Years", ylabel="Exposure", title="Simulated derivative counterparty exposure")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "counterparty_pfe.png", dpi=150)
    plt.close(fig)

    periods = 20
    common = dict(order_size=1_000_000, periods=periods, horizon=1., volatility=.3,
                  temporary_impact=1e-6, permanent_impact=1e-7)
    aggressive = OptimalExecution(risk_aversion=1e-4, **common).schedule()
    passive = OptimalExecution(risk_aversion=1e-7, **common).schedule()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, periods+1), aggressive.cumsum()/1_000_000, label="Aggressive")
    ax.plot(np.arange(1, periods+1), passive.cumsum()/1_000_000, label="Passive")
    ax.set(xlabel="Trading period", ylabel="Fraction liquidated", title="Almgren-Chriss liquidation")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "execution_schedule.png", dpi=150)
    plt.close(fig)

    lda = OpRiskLDA(annual_frequency=3., severity_family="lognormal",
                    severity_location=np.log(100_000), severity_shape=1.2).simulate(100_000, .999, 42)
    print(f"CVA: {exposure.cva:,.2f}")
    print(f"Operational expected loss: {lda.expected_loss:,.2f}")
    print(f"Operational VaR 99.9%: {lda.value_at_risk:,.2f}")


if __name__ == "__main__":
    main()
