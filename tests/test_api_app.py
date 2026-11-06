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
    monkeypatch.setattr(api_app, "_election_stress", lambda request, weights, returns: {"available": False})
    monkeypatch.setattr(api_app, "_purchase_dividends", lambda *args: 0.)
    return prices


def test_manual_portfolios_are_isolated(prices):
    first = api_app.portfolio_risk(api_app.PortfolioRequest(tickers=["aapl", "petr4.sa"], weights=[1, 0]))
    second = api_app.portfolio_risk(api_app.PortfolioRequest(tickers=["aapl", "petr4.sa"], weights=[0, 1]))
    assert first["weights"] == {"AAPL": 1., "PETR4.SA": 0.}
    assert second["weights"] == {"AAPL": 0., "PETR4.SA": 1.}
    assert first["wealth_index"] != second["wealth_index"]


def test_none_stress_skips_hypothetical_factors(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("Stress desativado não deve buscar fatores de mercado")
    monkeypatch.setattr(api_app, "_download_cached", unavailable)
    request = api_app.PortfolioRequest(tickers=["AAPL"], weights=[1.],
                                       stress_preset="none", stress_ibov=-.8)
    result = api_app._election_stress(request, pd.Series({"AAPL": 1.}),
                                      pd.DataFrame({"AAPL": [.01, -.01]}))
    assert result == {"available": False, "reason": "none"}
    with pytest.raises(ValueError):
        api_app.PortfolioRequest(tickers=["AAPL"], weights=[1.], stress_preset="invalid")


def test_liquidity_estimate_does_not_inherit_hypothetical_slippage(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=90, freq="B")
    history = pd.DataFrame({"AAPL": np.linspace(-.01, .01, len(dates))}, index=dates)
    monkeypatch.setattr(api_app, "_fx_cached", lambda *args: pd.Series([5.], index=[dates[-1]]))
    monkeypatch.setattr(api_app, "_volume_cached", lambda *args: 1_000_000.)
    first = api_app.PortfolioRequest(tickers=["AAPL"], weights=[1.], liquidity=True,
                                     stress_preset="none", stress_slippage_multiplier=1.)
    second = first.model_copy(update={"stress_slippage_multiplier": 8.})
    weights = pd.Series({"AAPL": 1.})
    base = api_app._liquidity_adjustment(first, weights, history, .02, .03)
    changed = api_app._liquidity_adjustment(second, weights, history, .02, .03)
    assert base == changed


@pytest.mark.parametrize("method", ["equal", "max_sharpe", "min_cvar", "risk_parity", "hrp"])
def test_optimizers(prices, method):
    risk = api_app.portfolio_risk(api_app.PortfolioRequest(tickers=["AAPL", "PETR4.SA"], method=method))
    assert np.isclose(sum(risk["weights"].values()), 1)
    assert all(weight >= 0 for weight in risk["weights"].values())
    if method == "equal":
        assert set(risk["weights"].values()) == {.5}


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
    assert lot["cost"] == 5000
    assert lot["market_value"] == 6000
    assert lot["pnl"] == 1000


def test_garch_forecast_and_out_of_sample():
    rng = np.random.default_rng(44)
    returns = rng.standard_t(7, 150) * .01
    var, es = api_app._model_var_es(returns, "garch")
    result = api_app._backtest(returns, pd.date_range("2025-01-01", periods=150, freq="B"), "garch")
    assert np.isfinite([var, es, result["p_value"]]).all()
    assert es >= var
    assert result["observations"] == 50


def test_benchmark_changes_with_base_currency(monkeypatch):
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    series = {"^BVSP": [100., 110., 120.], "^GSPC": [100., 110., 120.]}
    monkeypatch.setattr(api_app, "_download_cached", lambda tickers, *args:
                        pd.DataFrame({tickers[0]: series[tickers[0]]}, index=dates))
    monkeypatch.setattr(api_app, "_fx_cached", lambda *args:
                        pd.Series([5., 5.5, 6.], index=dates))
    for symbol in ("^BVSP", "^GSPC"):
        brl = api_app._benchmark_series(api_app.PortfolioRequest(
            tickers=["AAPL"], weights=[1.], benchmark=symbol, base_currency="BRL"), dates, 0)
        usd = api_app._benchmark_series(api_app.PortfolioRequest(
            tickers=["AAPL"], weights=[1.], benchmark=symbol, base_currency="USD"), dates, 0)
        assert brl["wealth_index"][-1] != usd["wealth_index"][-1]
        assert np.isclose(brl["wealth_index"][-1] / usd["wealth_index"][-1], 6 / 5)


def test_2008_replaces_2022_stress_window():
    assert any("2008" in label for label in api_app.STRESS_WINDOWS)
    assert all("2022" not in label for label in api_app.STRESS_WINDOWS)


def test_evt_tail_and_filtered_garch(monkeypatch):
    from quant_risk_engine.engine_market_tail.volatility import GarchFit
    rng = np.random.default_rng(120)
    series = rng.standard_t(5, 500)*.012
    series[0] = -.18
    monkeypatch.setattr(api_app, "fit_garch", lambda x: GarchFit(.02,.07,.89,0.,6.,0.,1.,-1.,100.))
    monkeypatch.setattr(api_app, "forecast_garch", lambda fit: (0., .012))
    for model in ("evt", "filtered_ewma_evt", "filtered_garch_evt"):
        scenarios, diagnostic = api_app._risk_scenarios(
            series, series[:, None], np.array([1.]), model, .90, 12000)
        var, es = api_app.historical_var_es(scenarios, .99)
        assert np.isfinite([var, es, diagnostic["shape"], diagnostic["ks_p_value"]]).all()
        assert es >= var and diagnostic["exceedances"] > 0


def test_student_t_copula_joint_tail():
    rng = np.random.default_rng(91)
    shared = rng.standard_t(5, 450)*.012
    assets = np.column_stack([shared+rng.normal(0,.003,450),
                              .7*shared+rng.normal(0,.005,450)])
    scenarios, diagnostic = api_app._risk_scenarios(
        assets @ np.array([.6,.4]), assets, np.array([.6,.4]),
        "copula_evt", .9, 12000)
    assert diagnostic["copula_nu"] > 2
    assert np.isclose(np.linalg.eigvalsh(diagnostic["copula_correlation"]).min() >= 0, True)
    assert np.isfinite(scenarios).all()


def test_christoffersen_and_consistent_budget():
    rng = np.random.default_rng(14)
    history = rng.normal(0,.02,400)
    dates = pd.date_range("2024-01-01", periods=400, freq="B")
    result = api_app._backtest(history, dates, "historical", .995)
    assert np.isfinite([result["p_ind"], result["p_cc"]]).all()
    assert np.isclose(result["lr_cc"], result["lr_pof"]+result["lr_ind"])
    scenarios = rng.normal(0,.025,10000)
    limits = {"var_99": .03, "es_99": .05, "volatility_annual": .5, "max_drawdown": .8}
    budget = api_app._risk_budget(history, pd.Series({"AAPL": 1.}), limits,
                                  np.full(400,.0004), scenarios, .995)
    assert np.isclose(budget["selected_metrics"]["var_99"],
                      np.quantile(-scenarios,.995))


def test_liquidity_and_election_stress(monkeypatch):
    rng = np.random.default_rng(13)
    dates = pd.date_range("2025-01-01", periods=160, freq="B")
    factors = pd.DataFrame({"^BVSP":100*np.cumprod(1+rng.normal(0,.01,160)),
                            "^GSPC":100*np.cumprod(1+rng.normal(0,.01,160))},index=dates)
    fx = pd.Series(5*np.cumprod(1+rng.normal(0,.002,160)),index=dates)
    asset = pd.DataFrame({"AAPL": .8*factors["^GSPC"].pct_change(fill_method=None).fillna(0)
                           +.3*fx.pct_change(fill_method=None).fillna(0)},index=dates)
    monkeypatch.setattr(api_app,"_download_cached",lambda *args: factors)
    monkeypatch.setattr(api_app,"_fx_cached",lambda *args: fx)
    monkeypatch.setattr(api_app,"_volume_cached",lambda *args: 1_000_000.)
    request = api_app.PortfolioRequest(tickers=["AAPL"],weights=[1.],liquidity=True,
                                       stress_preset="election")
    liquid = api_app._liquidity_adjustment(request,pd.Series({"AAPL":1.}),asset,.03,.05)
    stress = api_app._election_stress(request,pd.Series({"AAPL":1.}),asset)
    assert liquid["lvar"] > .03 and liquid["days"] >= 1
    assert np.isfinite([stress["return"],stress["pnl"]]).all()


def test_dividends_follow_split_and_base_currency(monkeypatch):
    dates = pd.to_datetime(["2025-01-15", "2025-02-15"])
    dividends = pd.Series([1., 2.], index=dates)
    splits = pd.Series([2.], index=pd.to_datetime(["2025-02-01"]))
    monkeypatch.setattr(api_app, "_dividend_events_cached", lambda *args: (dividends, splits))
    monkeypatch.setattr(api_app, "_fx_cached", lambda *args:
                        pd.Series([5., 6.], index=dates))
    total = api_app._purchase_dividends("AAPL", api_app.date(2025,1,1), 10., "BRL", 0)
    assert np.isclose(total, 290.)


def test_evt_api_json_and_limit_alignment(prices):
    result = api_app.portfolio_risk(api_app.PortfolioRequest(
        tickers=["AAPL", "PETR4.SA"], weights=[.6,.4], var_model="evt",
        confidence=.995, threshold_quantile=.90))
    json.dumps(result, allow_nan=False)
    assert result["risk_budget"]["model"] == "evt"
    assert np.isclose(result["metrics"]["var_99"],
                      result["risk_budget"]["selected_metrics"]["var_99"])


def test_evt_rolling_fit_keeps_expected_shortfall_finite(monkeypatch):
    from quant_risk_engine.engine_market_tail import evt
    original_fit = evt.genpareto.fit
    monkeypatch.setattr(evt.genpareto, "fit", lambda *args, **kwargs: (1.1, 0., .01))
    losses = np.random.default_rng(42).standard_t(4, 260) * .01
    fit = evt.fit_pot(losses, .9)
    var, es = evt.pot_var_es(fit, .995)
    assert fit.shape < .95 and np.isfinite([var, es]).all() and es >= var
    monkeypatch.setattr(evt.genpareto, "fit", original_fit)
