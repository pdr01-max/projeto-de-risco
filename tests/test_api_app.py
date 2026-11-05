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
    monkeypatch.setattr(api_app, "_download_cached", lambda tickers, period, bucket: prices)
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
