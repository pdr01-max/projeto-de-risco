import numpy as np
from quant_risk_engine.core.matrix import repair_psd
from quant_risk_engine.core.estimators import rolling_forecast
from quant_risk_engine.engine_market_tail import (
    normal_var_es, historical_var_es, pot_var_es, fit_pot,
    simulate_copula, fit_copula, kupiec_pof, christoffersen_independence,
)


def test_psd_and_copula():
    repaired = repair_psd([[1, 1.2], [1.2, 1]], correlation=True)
    assert np.linalg.eigvalsh(repaired).min() > 0
    assert np.allclose(np.diag(repaired), 1)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 2))
    for family in ("gaussian", "student_t"):
        u = simulate_copula(fit_copula(x, family), 1000)
        assert u.shape == (1000, 2) and np.all((u > 0) & (u < 1))


def test_risk_and_evt():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100_000)
    v, e = normal_var_es(0, 1, .99)
    hv, he = historical_var_es(x, .99)
    assert abs(v-hv) < .06 and abs(e-he) < .08
    assert e > v and he > hv
    pv, pe = pot_var_es(fit_pot(-x), .99)
    assert abs(pv-v) < .12 and pe > pv


def test_backtest_and_no_lookahead():
    b = np.zeros(100, dtype=int)
    b[::10] = 1
    assert kupiec_pof(b, .9).p_value > .5
    assert np.isfinite(christoffersen_independence(b).statistic)
    x = np.arange(20, dtype=float)
    prediction = rolling_forecast(x, 5, np.mean)
    assert prediction[0] == 2 and prediction[-1] == 16
