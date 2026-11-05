import numpy as np
from quant_risk_engine.module_counterparty import CounterpartyRisk
from quant_risk_engine.module_execution import OptimalExecution
from quant_risk_engine.module_systemic_ops import OpRiskLDA, SystemicRisk


def test_exposure_cva_exact():
    paths = [[10., -4.], [20., 6.]]
    result = CounterpartyRisk(paths, [1., 2.], [.01, .03], .4).calculate()
    assert np.allclose(result.expected_exposure, [15., 3.])
    assert np.isclose(result.cva, .6*(15*.01+3*.02))
    assert np.all(result.potential_future_exposure >= result.expected_exposure)


def test_execution_and_lda():
    common = dict(order_size=1000., periods=10, horizon=1., volatility=.3,
                  temporary_impact=1e-5)
    passive = OptimalExecution(risk_aversion=0., **common).schedule()
    aggressive = OptimalExecution(risk_aversion=.01, **common).schedule()
    assert np.allclose(passive, 100)
    assert np.isclose(aggressive.sum(), 1000)
    assert aggressive[0] > passive[0]
    result = OpRiskLDA(2., "lognormal", np.log(100.), .5).simulate(20_000, .99)
    assert result.value_at_risk > result.expected_loss > 0


def test_systemic_quantile_regression():
    rng = np.random.default_rng(3)
    institution = rng.normal(size=300)
    system = .7*institution+rng.normal(scale=.15, size=300)
    risk = SystemicRisk(institution, system)
    covar = risk.delta_covar(.05)
    assert .5 < covar.quantile_slope < .9
    assert covar.delta_covar > 0
    assert risk.marginal_expected_shortfall(.05) > 0
