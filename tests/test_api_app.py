import numpy as np
import pandas as pd
import pytest
import json
import api_app


@pytest.fixture
def prices(monkeypatch):
    rng = np.random.default_rng(12)
    returns = rng.normal(.0008, .01, (250, 2))
    prices = pd.DataFrame(100*np.cumprod(1+returns, axis=0),
                          index=pd.date_range("2025-01-01", periods=250, freq="B"),
                          columns=["AAPL", "PETR4.SA"])
    monkeypatch.setattr(api_app, "_download_cached", lambda tickers, period, bucket, start=None, end=None: prices)
    monkeypatch.setattr(api_app, "_fx_cached", lambda start, end, bucket: pd.Series(5., index=prices.index))
    monkeypatch.setattr(api_app, "_cash_returns", lambda returns, tickers: np.full(len(returns), .0004))
    monkeypatch.setattr(api_app, "_stress_tests", lambda request, weights, bucket: [])
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


def test_brl_conversion_and_strict_calendar(monkeypatch):
    dates = pd.date_range("2026-01-05", periods=83, freq="B")
    prices = pd.DataFrame({"AAPL": 100 + np.arange(83),
                           "PETR4.SA": 30 + np.arange(83)/10}, index=dates)
    prices.loc[dates[20], "PETR4.SA"] = np.nan
    monkeypatch.setattr(api_app, "_download_cached", lambda *args: prices)
    monkeypatch.setattr(api_app, "_fx_cached", lambda *args: pd.Series(5., index=dates))
    request = api_app.PortfolioRequest(tickers=["AAPL", "PETR4.SA"], weights=[.5, .5])
    returns = api_app._returns(request)
    assert len(returns) == 81
    assert dates[20] not in returns.index
    assert np.isclose(returns.loc[dates[1], "AAPL"], 505/500-1)


def test_model_backtest_and_json(prices):
    for model in ("historical", "normal", "ewma", "monte_carlo"):
        result = api_app.portfolio_risk(api_app.PortfolioRequest(
            tickers=["AAPL", "PETR4.SA"], weights=[.5, .5], var_model=model,
            base_currency="BRL", capital=100000))
        assert result["backtest"]["observations"] > 0
        assert np.isclose(sum(result["risk_contribution"].values()), 1)
        json.dumps(result, allow_nan=False)


def test_backtest_forecasts_do_not_see_realized_day():
    rng = np.random.default_rng(88)
    returns = rng.normal(0, .01, 300)
    dates = pd.date_range("2025-01-01", periods=300, freq="B")
    first = api_app._backtest(returns, dates, "historical")
    returns[-1] = -.5
    second = api_app._backtest(returns, dates, "historical")
    assert first["forecast_var"] == second["forecast_var"]
    assert second["breaches"] == first["breaches"] + 1


def test_purchase_split_adjusts_share_count(prices, monkeypatch):
    monkeypatch.setattr(api_app, "_latest_close_cached", lambda *args: {"AAPL": 60.})
    monkeypatch.setattr(api_app, "_splits_cached", lambda *args: 2.)
    result = api_app.portfolio_risk(api_app.PortfolioRequest(
        tickers=["AAPL", "PETR4.SA"], weights=[1., 0.],
        purchases={"AAPL": {"date": "2025-01-10", "price": 100., "quantity": 10.}}))
    lot = result["purchase_analysis"][0]
    assert lot["adjusted_quantity"] == 20
    assert lot["cost"] == 1000
    assert lot["market_value"] == 1200
    assert lot["pnl"] == 200


def test_garch_forecast_and_out_of_sample():
    rng = np.random.default_rng(44)
    returns = rng.standard_t(7, 150) * .01
    var, es = api_app._model_var_es(returns, "garch")
    result = api_app._backtest(returns, pd.date_range("2025-01-01", periods=150, freq="B"), "garch")
    assert np.isfinite([var, es, result["p_value"]]).all()
    assert es >= var
    assert result["observations"] == 50
