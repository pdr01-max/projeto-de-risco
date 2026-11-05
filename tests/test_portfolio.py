import numpy as np
import pandas as pd
from quant_risk_engine.engine_portfolio import (
    MeanVarianceOptimizer, MinCVaROptimizer, RiskParityOptimizer,
    HRPOptimizer, BlackLitterman, FractionalKelly,
)


def test_all_portfolios_sum_to_one():
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(300, 3))*.01 + [.0004, .0002, .0003], columns=list("ABC"))
    covariance = np.cov(x, rowvar=False)
    portfolios = [
        MeanVarianceOptimizer().optimize(x),
        MinCVaROptimizer().optimize(x),
        RiskParityOptimizer().optimize(x),
        HRPOptimizer().optimize(x),
        BlackLitterman().optimize(covariance, np.ones(3)/3, [[1, -1, 0]], [.0001]),
        FractionalKelly().optimize(x.mean(), covariance),
    ]
    for weights in portfolios:
        assert np.isclose(weights.sum(), 1, atol=1e-6)
        assert (weights >= -1e-9).all()


def test_risk_parity_contributions():
    covariance = np.diag([.01, .04, .09])
    weights = RiskParityOptimizer().optimize(covariance=covariance).to_numpy()
    contributions = weights*(covariance@weights)
    assert np.max(contributions)-np.min(contributions) < 1e-5
