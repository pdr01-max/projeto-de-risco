import numpy as np
from quant_risk_engine.module_credit import (
    MertonModel, VasicekIRB, CreditMetrics, altman_z_public, altman_z_private,
)


def test_merton_calibration():
    result = MertonModel(30., .75, 100., 1., .04, .03).solve()
    assert result.asset_value > 30
    assert result.asset_volatility > 0
    assert 0 < result.probability_of_default < 1
    assert max(abs(result.equity_residual), abs(result.volatility_residual)) < 1e-7


def test_irb_vectorization_and_scaling():
    result = VasicekIRB([.01, .03], [.4, .5], [100., 200.], [2.5, 2.5]).calculate()
    doubled = VasicekIRB([.01, .03], [.4, .5], [200., 400.], [2.5, 2.5]).calculate()
    assert np.allclose(result.expected_loss, [.4, 3.])
    assert np.allclose(doubled.risk_weighted_assets, 2*result.risk_weighted_assets)
    assert np.allclose(result.risk_weighted_assets, 12.5*result.unexpected_loss_capital)


def test_markov_and_z_scores():
    transition = [[.9, .08, .02], [.0, .85, .15], [0, 0, 1]]
    metrics = CreditMetrics(transition, ("A", "B", "D"))
    p2 = metrics.cumulative_default_probability(2)
    assert np.isclose(p2["A"], .9*.02+.08*.15+.02)
    assert np.isclose(p2["D"], 1)
    public = altman_z_public(20., 10., 12., 80., 40., 100., 120.)
    private = altman_z_private(20., 10., 12., 60., 40., 100., 120.)
    assert np.isfinite(public) and np.isfinite(private)
