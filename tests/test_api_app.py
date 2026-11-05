import numpy as np
import pandas as pd
import pytest
import api_app


@pytest.fixture
def prices(monkeypatch):
    rng = np.random.default_rng(12)
    returns = rng.normal(.0008, .01, (250, 2))
    prices = pd.DataFrame(100*np.cumprod(1+returns, axis=0),
                          index=pd.date_range("2025-01-01", periods=250, freq="B"),
                          columns=["AAPL", "PETR4.SA"])
    monkeypatch.setattr(api_app, "_download_cached", lambda tickers, period, bucket, start=None, end=None: prices)
    monkeypatch.setattr(api_app, "_cash_returns", lambda returns, tickers: np.full(len(returns), .0004))
    return prices


def test_manual_portfolios_are_isolated(prices):
    first = api_app.portfolio_risk(api_app.PortfolioRequest(tickers=["aapl", "petr4.sa"], weights=[1, 0]))
    second = api_app.portfolio_risk(api_app.PortfolioRequest(tickers=["aapl", "petr4.sa"], weights=[0, 1]))
    assert first["weights"] == {"AAPL": 1., "PETR4.SA": 0.}
    assert second["weights"] == {"AAPL": 0., "PETR4.SA": 1.}
    assert first["wealth_index"] != second["wealth_index"]


@pytest.mark.parametrize("method", ["max_sharpe", "min_cvar", "risk_parity", "hrp"])
def test_optimizers(prices, method):
    risk = api_app.portfolio_risk(api_app.PortfolioRequest(tickers=["AAPL", "PETR4.SA"], method=method))
    assert np.isclose(sum(risk["weights"].values()), 1)
    assert all(weight >= 0 for weight in risk["weights"].values())


def test_invalid_weights_and_static_route(prices):
    with pytest.raises(ValueError):
        api_app.PortfolioRequest(tickers=["AAPL", "PETR4.SA"], weights=[.4, .4])
    assert str(api_app.index().path).endswith("index.html")
    assert api_app.health() == {"status": "ok"}


def test_rebalance_respects_all_limits():
    rng = np.random.default_rng(31)
    portfolio = rng.normal(.0003, .025, 500)
    weights = pd.Series({"AAPL": .7, "SPY": .3})
    limits = {"var_99": .015, "es_99": .025,
              "volatility_annual": .12, "max_drawdown": .08}
    cash = np.full(len(portfolio), .0004)
    result = api_app._risk_budget(portfolio, weights, limits, cash)
    assert result["breaches"]
    assert result["compliant"]
    assert np.isclose(sum(result["recommended_weights"].values()), 1)
    assert all(result["recommended_metrics"][key] <= limit + 1e-8
               for key, limit in limits.items())


def test_cdi_cash_changes_risk_and_return():
    risky = np.tile([-.04, .015, .01, .005], 100)
    cash = np.full(len(risky), .0005)
    limits = {"var_99": .02, "es_99": .025,
              "volatility_annual": .20, "max_drawdown": .20}
    result = api_app._risk_budget(risky, pd.Series({"PETR4.SA": 1.}), limits, cash)
    scale = result["risk_scale"]
    expected = cash + scale * (risky - cash)
    assert result["compliant"]
    assert np.isclose(result["recommended_return_annual"],
                      np.prod(1 + expected) ** (252 / len(expected)) - 1)
    assert result["cash_return_annual"] > 0


def test_unattainable_cash_limit_is_reported():
    risky = np.tile([-.05, .01], 100)
    cash = np.tile([-.04, .01], 100)
    limits = {"var_99": .001, "es_99": .001,
              "volatility_annual": .01, "max_drawdown": .01}
    result = api_app._risk_budget(risky, pd.Series({"AAPL": 1.}), limits, cash)
    assert not result["compliant"]
