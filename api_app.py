"""Public dashboard: uvicorn api_app:app --host 0.0.0.0 --port 7860."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
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
    start_date: date | None = None
    end_date: date | None = None
    purchases: dict[str, dict] | None = None
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
        if self.period not in {"6mo", "1y", "2y", "5y", "10y", "max", "custom"} or self.method not in METHODS:
            raise ValueError("Período ou método de alocação inválido")
        if self.period == "custom":
            if not self.start_date or not self.end_date or self.start_date >= self.end_date or self.end_date > date.today():
                raise ValueError("Informe datas inicial e final válidas, até hoje.")
        if self.purchases:
            if set(self.purchases) - set(self.tickers):
                raise ValueError("Compra informada para ativo fora da carteira.")
            for symbol, purchase in self.purchases.items():
                try:
                    bought = date.fromisoformat(str(purchase["date"]))
                    price = float(purchase["price"])
                    quantity = float(purchase["quantity"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Compra inválida para {symbol}.") from exc
                if bought > date.today() or not np.isfinite([price, quantity]).all() or price <= 0 or quantity <= 0:
                    raise ValueError(f"Data, preço e quantidade de {symbol} devem ser válidos e positivos.")
        if self.method == "manual":
            if self.weights is None or len(self.weights) != len(self.tickers):
                raise ValueError("Informe um peso para cada ativo")
            weights = np.asarray(self.weights, float)
            if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-5):
                raise ValueError("Pesos manuais devem ser não negativos e somar 100%")
        return self


@lru_cache(maxsize=128)
def _download_cached(tickers: tuple[str, ...], period: str, bucket: int,
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Cache market-only prices for 15 minutes; never cache portfolios or limits."""
    try:
        params = {"start": start, "end": end} if start and end else {"period": period}
        raw = yf.download(list(tickers), **params, interval="1d", auto_adjust=True,
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
            fx_raw = yf.download("USDBRL=X", **params, interval="1d", auto_adjust=True,
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
    end = (request.end_date + timedelta(days=1)).isoformat() if request.end_date else None
    prices = _download_cached(tuple(sorted(request.tickers)), request.period, int(time.time() // 900),
                              request.start_date.isoformat() if request.start_date else None, end)
    returns = prices.loc[:, request.tickers].pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise HTTPException(422, "Retornos inválidos; tente outra janela histórica.")
    returns.attrs["first_price_date"] = prices.index[0]
    return returns


@lru_cache(maxsize=64)
def _cdi_cached(start: date, end: date, bucket: int) -> pd.Series:
    """BCB SGS 12: daily CDI in percent; historical requests are split below 10 years."""
    chunks: list[pd.Series] = []
    left = start
    try:
        while left <= end:
            right = min(left + timedelta(days=365 * 8), end)
            query = urlencode({"formato": "json", "dataInicial": left.strftime("%d/%m/%Y"),
                               "dataFinal": right.strftime("%d/%m/%Y")})
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?{query}"
            with urlopen(Request(url, headers={"User-Agent": "quant-risk-engine/2.0"}), timeout=20) as response:
                observations = json.load(response)
            if observations:
                frame = pd.DataFrame(observations)
                values = pd.Series(pd.to_numeric(frame["valor"].str.replace(",", "."), errors="raise").to_numpy() / 100,
                                   index=pd.to_datetime(frame["data"], format="%d/%m/%Y"))
                chunks.append(values)
            left = right + timedelta(days=1)
    except Exception as exc:
        raise HTTPException(502, "CDI histórico indisponível no Banco Central. Tente novamente em instantes.") from exc
    if not chunks:
        raise HTTPException(502, "Sem observações de CDI para a janela selecionada.")
    cdi = pd.concat(chunks).sort_index()
    if not np.isfinite(cdi.to_numpy()).all() or (cdi < 0).any():
        raise HTTPException(502, "Série histórica de CDI inválida.")
    return cdi


@lru_cache(maxsize=64)
def _fx_cached(start: date, end: date, bucket: int) -> pd.Series:
    """USD/BRL adjusted close for valuing CDI cash in a USD portfolio."""
    try:
        raw = yf.download("USDBRL=X", start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                          interval="1d", auto_adjust=True, progress=False, threads=False, timeout=15)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().sort_index()
        if close.empty or (close <= 0).any():
            raise ValueError("missing FX")
        return close
    except Exception as exc:
        raise HTTPException(502, "USD/BRL indisponível para converter o caixa CDI em dólares.") from exc


def _cash_returns(returns: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """Cash levels grow by daily CDI in BRL; USD portfolios divide by USD/BRL."""
    previous = pd.Timestamp(returns.attrs["first_price_date"])
    dates = pd.DatetimeIndex([previous]).append(returns.index)
    cdi = _cdi_cached(previous.date(), dates[-1].date(), int(time.time() // 86400))
    daily = cdi.reindex(pd.date_range(previous, dates[-1], freq="D"), fill_value=0.)
    levels = (1 + daily).cumprod().reindex(dates).to_numpy()
    if any(not name.endswith(".SA") for name in tickers):
        fx = _fx_cached((previous - timedelta(days=7)).date(), dates[-1].date(), int(time.time() // 900))
        aligned = fx.reindex(dates, method="ffill")
        if aligned.isna().any():
            raise HTTPException(502, "USD/BRL sem cobertura para toda a janela do caixa CDI.")
        levels = levels / aligned.to_numpy()
    cash = levels[1:] / levels[:-1] - 1
    if not np.isfinite(cash).all() or np.any(cash <= -1):
        raise HTTPException(502, "Retornos inválidos do caixa CDI.")
    return cash


@lru_cache(maxsize=128)
def _latest_close_cached(tickers: tuple[str, ...], bucket: int) -> dict[str, float]:
    """Current unadjusted closes, in each asset's quote currency, for cost-basis comparison."""
    raw = yf.download(list(tickers), period="5d", interval="1d", auto_adjust=False,
                      progress=False, threads=False, timeout=15)
    if raw.empty:
        raise HTTPException(502, "Não foi possível obter preços atuais para avaliar as compras.")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    if len(tickers) == 1:
        close.columns = [tickers[0]]
    result = {}
    for symbol in tickers:
        if symbol not in close or close[symbol].dropna().empty:
            raise HTTPException(422, f"Sem preço atual para {symbol}.")
        result[symbol] = float(close[symbol].dropna().iloc[-1])
    return result


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


def _budget_metrics(portfolio: np.ndarray) -> dict[str, float]:
    """Historical VaR/ES and compounded drawdown for a fixed allocation."""
    wealth = np.cumprod(1 + portfolio)
    drawdown = wealth / np.maximum.accumulate(np.r_[1., wealth])[1:] - 1
    var, es = historical_var_es(portfolio, .99)
    return {"var_99": float(var), "es_99": float(es),
            "volatility_annual": float(portfolio.std(ddof=1) * np.sqrt(252)),
            "max_drawdown": float(-drawdown.min())}


def _risk_budget(portfolio: np.ndarray, weights: pd.Series,
                 limits: dict[str, float], cash_returns: np.ndarray | None = None) -> dict:
    """R(a)=a*R_risky+(1-a)*R_CDI; evaluate VaR, ES, vol and drawdown on each a."""
    selected = _budget_metrics(portfolio)
    breaches = [{"metric": key, "actual": selected[key], "limit": limit}
                for key, limit in limits.items() if selected[key] > limit + 1e-10]
    scale = 1.
    after = selected
    if breaches:
        if cash_returns is None:
            raise ValueError("CDI cash returns are required for a breached allocation")
        candidates = np.linspace(0, 1, 1001)
        scenarios = cash_returns[:, None] + (portfolio - cash_returns)[:, None] * candidates[None, :]
        loss = -scenarios
        var = np.quantile(loss, .99, axis=0)
        es = np.sum(np.where(loss >= var[None, :], loss, 0.), axis=0) / np.sum(loss >= var[None, :], axis=0)
        vol = scenarios.std(axis=0, ddof=1) * np.sqrt(252)
        paths = np.cumprod(1 + scenarios, axis=0)
        peaks = np.maximum.accumulate(np.vstack([np.ones((1, len(candidates))), paths]), axis=0)[1:]
        drawdowns = 1 - paths / peaks
        feasible = np.flatnonzero((var <= limits["var_99"]) & (es <= limits["es_99"])
                                  & (vol <= limits["volatility_annual"])
                                  & (drawdowns.max(axis=0) <= limits["max_drawdown"]))
        scale = float(candidates[feasible[-1]]) if len(feasible) else 0.
        after = _budget_metrics(cash_returns + (portfolio - cash_returns) * scale)
    recommended = {name: float(value * scale) for name, value in weights.items()}
    recommended["CAIXA"] = float(1 - scale)
    return {"breaches": breaches,
            "selected_metrics": selected, "recommended_metrics": after,
            "recommended_weights": recommended, "risk_scale": scale,
            "cash_return_annual": (float(np.prod(1 + cash_returns) ** (252 / len(cash_returns)) - 1)
                                   if cash_returns is not None else None),
            "recommended_return_annual": float(np.prod(1 + (cash_returns + (portfolio - cash_returns) * scale)
                                                      if cash_returns is not None else (1 + portfolio))
                                               ** (252 / len(portfolio)) - 1),
            "compliant": all(after[key] <= limit + 1e-8 for key, limit in limits.items())}


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
    metrics = _budget_metrics(portfolio)
    rolling_vol = pd.Series(portfolio).rolling(21, min_periods=21).std().mul(np.sqrt(252))
    covariance = returns.cov().to_numpy()
    w = weights.loc[request.tickers].to_numpy()
    portfolio_vol = float(np.sqrt(w @ covariance @ w))
    risk_contribution = (np.sqrt(252) * w * (covariance @ w) / portfolio_vol
                         if portfolio_vol > 0 else np.zeros_like(w))
    correlation = returns.corr().fillna(0).to_numpy()
    np.fill_diagonal(correlation, 1.)
    purchase_analysis = []
    if request.purchases:
        current = _latest_close_cached(tuple(sorted(request.purchases)), int(time.time() // 900))
        for symbol, purchase in request.purchases.items():
            price, quantity = float(purchase["price"]), float(purchase["quantity"])
            latest = current[symbol]
            holding_days = (date.today() - date.fromisoformat(str(purchase["date"]))).days
            purchase_analysis.append({
                "ticker": symbol, "date": str(purchase["date"]),
                "currency": "BRL" if symbol.endswith(".SA") else "USD",
                "quantity": quantity, "purchase_price": price, "current_price": latest,
                "cost": price * quantity, "market_value": latest * quantity,
                "pnl": (latest - price) * quantity, "return": latest / price - 1,
                "holding_days": holding_days,
                "return_annualized": (latest / price) ** (365.25 / holding_days) - 1
                                     if holding_days > 0 else None,
            })
    metrics.update({
        "return_annual": float(np.prod(1 + portfolio) ** (252 / len(portfolio)) - 1),
        "downside_volatility_annual": float(np.sqrt(np.mean(np.minimum(portfolio, 0) ** 2) * 252)),
        "best_day": float(portfolio.max()),
        "worst_day": float(portfolio.min()),
        "var_95": float(historical_var_es(portfolio, .95)[0]),
    })
    limits = {"var_99": request.var_limit, "es_99": request.es_limit,
              "volatility_annual": request.volatility_limit, "max_drawdown": request.drawdown_limit}
    cash_returns = _cash_returns(returns, request.tickers) if any(
        metrics[key] > limit + 1e-10 for key, limit in limits.items()) else None
    risk_budget = _risk_budget(portfolio, weights, limits, cash_returns)
    risk_budget["selected_method"] = request.method
    return {"tickers": request.tickers, "method": request.method, "weights": weights.to_dict(),
            "metrics": metrics, "limits": limits, "dates": returns.index.strftime("%Y-%m-%d").tolist(),
            "wealth_index": wealth.tolist(), "daily_losses": (-portfolio).tolist(),
            "drawdown": drawdown.tolist(),
            "rolling_volatility_21d": [None if pd.isna(v) else float(v) for v in rolling_vol],
            "correlation": correlation.tolist(),
            "risk_contribution": dict(zip(request.tickers, risk_contribution.tolist())),
            "purchase_analysis": purchase_analysis,
            "risk_budget": risk_budget,
            "observations": len(returns)}


@app.post("/api/optimize")
def optimize(request: PortfolioRequest) -> dict:
    returns = _returns(request)
    methods = ("max_sharpe", "min_cvar", "risk_parity", "hrp")
    allocations = {name: _optimize(returns, name).to_dict() for name in methods}
    return {"tickers": request.tickers, "allocations": allocations, "observations": len(returns)}
