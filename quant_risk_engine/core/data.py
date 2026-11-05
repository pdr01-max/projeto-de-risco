"""Market data ingestion and return transformations."""
import numpy as np
import pandas as pd


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Require positive finite adjusted prices with strictly increasing dates."""
    if not isinstance(prices, pd.DataFrame) or prices.empty or prices.shape[1] == 0:
        raise ValueError("prices must be a nonempty DataFrame")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("index must be unique and increasing")
    result = prices.astype(float)
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError("prices must be finite and strictly positive")
    return result


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute r[t] = log(P[t]/P[t-1]); prices must be adjusted for corporate actions."""
    p = validate_prices(prices)
    if len(p) < 2:
        raise ValueError("at least two observations required")
    return np.log(p).diff().iloc[1:]
