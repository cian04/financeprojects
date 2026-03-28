"""
metrics.py — Performance metrics for FX carry backtest.
All metrics are annualised unless stated otherwise.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def sharpe_ratio(returns: pd.Series, rf: float = 0.02) -> float:
    excess = returns - rf / TRADING_DAYS
    if excess.std() == 0:
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())

def sortino_ratio(returns: pd.Series, rf: float = 0.02) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside)

def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())

def calmar_ratio(returns: pd.Series) -> float:
    ann_return = annualised_return(returns)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(ann_return / mdd)

def annualised_return(returns: pd.Series) -> float:
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return float(total ** (1 / n_years) - 1)

def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))

def hit_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())

def summary(returns: pd.Series, label: str = "Strategy", rf: float = 0.02) -> pd.Series:
    return pd.Series({
        "Ann. Return":     f"{annualised_return(returns):.2%}",
        "Ann. Volatility": f"{annualised_volatility(returns):.2%}",
        "Sharpe Ratio":    f"{sharpe_ratio(returns, rf):.2f}",
        "Sortino Ratio":   f"{sortino_ratio(returns, rf):.2f}",
        "Max Drawdown":    f"{max_drawdown(returns):.2%}",
        "Calmar Ratio":    f"{calmar_ratio(returns):.2f}",
        "Hit Rate":        f"{hit_rate(returns):.2%}",
        "Observations":    len(returns),
    }, name=label)
