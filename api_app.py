"""Public dashboard: uvicorn api_app:app --host 0.0.0.0 --port 7860."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import time
import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from quant_risk_engine.core.estimators import ledoit_wolf_covariance
from quant_risk_engine.engine_market_tail.risk import historical_var_es
from quant_risk_engine.engine_portfolio import MinCVaROptimizer, RiskParityOptimizer, HRPOptimizer

STATIC = Path(__file__).resolve().parent / "static"
TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.^=-]{0,19}$")
METHODS = {"manual", "max_sharpe", "min_cvar", "risk_parity", "hrp"}
app = FastAPI(title="Quant Risk Engine", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class PortfolioRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=12)
    period: str = "2y"
    method: str = "manual"
    weights: list[float] | None = None  # fractions, not percentages
    var_limit: float = Field(default=.03, gt=0, lt=1)
    es_limit: float = Field(default=.05, gt=0, lt=1)
    volatility_limit: float = Field(default=.25, gt=0, lt=5)
    drawdown_limit: float = Field(default=.20, gt=0, lt=1)

    @field_validator("tickers")
    @classmethod
    def clean_tickers(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip().upper() for v in values]
        if len(set(cleaned)) != len(cleaned) or any(not TICKER.fullmatch(v) for v in cleaned):
            raise ValueError("Use tickers válidos e sem duplicatas, como PETR4.SA ou AAPL")
        return cleaned

    @model_validator(mode="after")
    def validate_portfolio(self) -> PortfolioRequest:
        if self.period not in {"6mo", "1y", "2y", "5y"} or self.method not in METHODS:
            raise ValueError("Período ou método de alocação inválido")
        if self.method == "manual":
            if self.weights is None or len(self.weights) != len(self.tickers):
                raise ValueError("Informe um peso para cada ativo")
            weights = np.asarray(self.weights, float)
            if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-5):
                raise ValueError("Pesos manuais devem ser não negativos e somar 100%")
        return self


@lru_cache(maxsize=128)
def _download_cached(tickers: tuple[str, ...], period: str, bucket: int) -> pd.DataFrame:
    """Cache market-only prices for 15 minutes; never cache portfolios or limits."""
    try:
        raw = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True,
                          progress=False, threads=False, timeout=15)
    except Exception as exc:
        raise HTTPException(502, "Provedor de preços indisponível. Tente novamente em instantes.") from exc
    if raw.empty:
        raise HTTPException(502, "Não foi possível obter preços. Verifique os tickers e tente novamente.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    if len(tickers) == 1:
        close.columns = [tickers[0]]
    missing = [symbol for symbol in tickers if symbol not in close or close[symbol].isna().all()]
    if missing:
        raise HTTPException(422, f"Sem histórico para: {', '.join(missing)}. Confira os símbolos; B3 requer .SA.")
    close = close.loc[:, list(tickers)].sort_index().ffill().dropna(how="any")
    if any(symbol.endswith(".SA") for symbol in tickers) and any(not symbol.endswith(".SA") for symbol in tickers):
        try:
            fx_raw = yf.download("USDBRL=X", period=period, interval="1d", auto_adjust=True,
                                 progress=False, threads=False, timeout=15)
            fx = fx_raw["Close"]
            if isinstance(fx, pd.DataFrame):
                fx = fx.iloc[:, 0]
            fx = fx.reindex(close.index).ffill()
            if fx.isna().any() or (fx <= 0).any():
                raise ValueError("FX unavailable")
        except Exception as exc:
            raise HTTPException(502, "Câmbio USDBRL indisponível para combinar ativos B3 e EUA.") from exc
        close.loc[:, [s for s in tickers if s.endswith(".SA")]] = close.loc[:, [s for s in tickers if s.endswith(".SA")]].div(fx, axis=0)
    if len(close) < 80 or not np.isfinite(close.to_numpy()).all() or (close <= 0).any().any():
        raise HTTPException(422, "Histórico comum insuficiente após alinhar feriados de Brasil e EUA.")
    return close


def _returns(request: PortfolioRequest) -> pd.DataFrame:
    prices = _download_cached(tuple(sorted(request.tickers)), request.period, int(time.time() // 900))
    returns = prices.loc[:, request.tickers].pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise HTTPException(422, "Retornos inválidos; tente outra janela histórica.")
    return returns


def _max_sharpe(returns: pd.DataFrame) -> pd.Series:
    """Long-only max Sharpe via homogeneous convex min variance: min x'Sigma x, mu'x>=1."""
    mu = returns.mean().to_numpy()
    if np.max(mu) <= 0:
        raise HTTPException(422, "Max Sharpe requer ao menos um retorno médio positivo nesta janela.")
    sigma = ledoit_wolf_covariance(returns.to_numpy())
    x = cp.Variable(len(mu), nonneg=True)
    scaled_mean = mu / np.max(mu)
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, sigma)), [scaled_mean @ x >= 1])
    problem.solve()
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        raise HTTPException(422, "Não foi possível encontrar uma carteira Max Sharpe.")
    weights = np.maximum(np.asarray(x.value).ravel(), 0)
    return pd.Series(weights / weights.sum(), index=returns.columns)


def _optimize(returns: pd.DataFrame, method: str) -> pd.Series:
    if returns.shape[1] == 1:
        return pd.Series([1.], index=returns.columns)
    try:
        if method == "max_sharpe":
            return _max_sharpe(returns)
        if method == "min_cvar":
            return MinCVaROptimizer(alpha=.99).optimize(returns)
        covariance = ledoit_wolf_covariance(returns.to_numpy())
        if method == "risk_parity":
            return RiskParityOptimizer().optimize(returns, covariance=covariance)
        if method == "hrp":
            return HRPOptimizer().optimize(returns, covariance=covariance)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, f"Otimização {method} falhou: {exc}") from exc
    raise HTTPException(422, "Método de otimização inválido")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/portfolio_risk")
def portfolio_risk(request: PortfolioRequest) -> dict:
    returns = _returns(request)
    weights = (pd.Series(request.weights, index=request.tickers) if request.method == "manual"
               else _optimize(returns, request.method))
    portfolio = returns.to_numpy() @ weights.loc[request.tickers].to_numpy()
    wealth = np.cumprod(1 + portfolio)
    drawdown = wealth / np.maximum.accumulate(np.r_[1., wealth])[1:] - 1
    var, es = historical_var_es(portfolio, .99)
    metrics = {"var_99": float(var), "es_99": float(es),
               "volatility_annual": float(portfolio.std(ddof=1) * np.sqrt(252)),
               "max_drawdown": float(-drawdown.min())}
    rolling_vol = pd.Series(portfolio).rolling(21, min_periods=21).std().mul(np.sqrt(252))
    covariance = returns.cov().to_numpy()
    w = weights.loc[request.tickers].to_numpy()
    portfolio_vol = float(np.sqrt(w @ covariance @ w))
    risk_contribution = (np.sqrt(252) * w * (covariance @ w) / portfolio_vol
                         if portfolio_vol > 0 else np.zeros_like(w))
    correlation = returns.corr().fillna(0).to_numpy()
    np.fill_diagonal(correlation, 1.)
    metrics.update({
        "return_annual": float(np.prod(1 + portfolio) ** (252 / len(portfolio)) - 1),
        "downside_volatility_annual": float(np.sqrt(np.mean(np.minimum(portfolio, 0) ** 2) * 252)),
        "best_day": float(portfolio.max()),
        "worst_day": float(portfolio.min()),
        "var_95": float(historical_var_es(portfolio, .95)[0]),
    })
    limits = {"var_99": request.var_limit, "es_99": request.es_limit,
              "volatility_annual": request.volatility_limit, "max_drawdown": request.drawdown_limit}
    return {"tickers": request.tickers, "method": request.method, "weights": weights.to_dict(),
            "metrics": metrics, "limits": limits, "dates": returns.index.strftime("%Y-%m-%d").tolist(),
            "wealth_index": wealth.tolist(), "daily_losses": (-portfolio).tolist(),
            "drawdown": drawdown.tolist(),
            "rolling_volatility_21d": [None if pd.isna(v) else float(v) for v in rolling_vol],
            "correlation": correlation.tolist(),
            "risk_contribution": dict(zip(request.tickers, risk_contribution.tolist())),
            "observations": len(returns)}


@app.post("/api/optimize")
def optimize(request: PortfolioRequest) -> dict:
    returns = _returns(request)
    methods = ("max_sharpe", "min_cvar", "risk_parity", "hrp")
    allocations = {name: _optimize(returns, name).to_dict() for name in methods}
    return {"tickers": request.tickers, "allocations": allocations, "observations": len(returns)}
