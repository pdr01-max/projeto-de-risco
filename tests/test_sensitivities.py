import numpy as np
from quant_risk_engine.engine_sensitivities import BlackScholesEngine, YieldCurveRisk


def test_option_parity_and_analytic_greeks():
    engine = BlackScholesEngine(np.array([90., 100., 110.]), 100., 1., .04, .2, .01)
    call, put = engine.price("call"), engine.price("put")
    assert np.allclose(call-put, np.array([90., 100., 110.])*np.exp(-.01)-100*np.exp(-.04))
    assert np.allclose(engine.delta("call")-engine.delta("put"), np.exp(-.01))
    assert np.all(engine.gamma() > 0)
    assert np.all(engine.vega() > 0)
    assert np.isfinite(engine.vanna()).all()


def test_bond_analytic_shock_and_key_rates():
    times = np.array([1., 2., 3.])
    cashflows = np.array([[5., 5., 105.], [0., 100., 0.]])
    curve = YieldCurveRisk(cashflows, times, np.array([.03, .04, .05]), 2)
    price = curve.price()
    duration = curve.modified_duration()
    convexity = curve.effective_convexity()
    key_rates = curve.key_rate_duration([1., 2., 3.])
    assert np.allclose(key_rates.sum(axis=1), duration)
    assert np.allclose(curve.pv01(), price*duration*1e-4)
    shifted = YieldCurveRisk(cashflows, times, np.array([.031, .041, .051]), 2).price()
    approximation = price*(1-duration*.001+.5*convexity*.001**2)
    assert np.allclose(shifted, approximation, atol=1e-6)
