"""
portfolio.py — Position sizing and portfolio construction for FX carry strategy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def vol_target_scalar(
    portfolio_returns: pd.Series,
    target_vol: float = 0.08,
    window: int = 21,
    min_scalar: float = 0.5,
    max_scalar: float = 3.0,
) -> pd.Series:
    """Compute daily vol-targeting scalar. Shifted 1 day to avoid look-ahead bias."""
    realised_vol = portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    scalar = (target_vol / realised_vol).clip(min_scalar, max_scalar)
    return scalar.shift(1).fillna(1.0)


def apply_vol_targeting(
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    target_vol: float = 0.08,
    window: int = 21,
    min_scalar: float = 0.5,
    max_scalar: float = 3.0,
) -> pd.DataFrame:
    """Scale weights so expected portfolio vol = target_vol."""
    port_ret_raw = (weights * log_returns).sum(axis=1)
    scalar = vol_target_scalar(port_ret_raw, target_vol, window, min_scalar, max_scalar)
    return weights.multiply(scalar, axis=0)


def apply_position_limits(
    weights: pd.DataFrame,
    max_weight: float = 0.4,
) -> pd.DataFrame:
    """Clip individual pair weights to [-max_weight, +max_weight] and re-normalise."""
    clipped = weights.clip(-max_weight, max_weight)
    abs_sum = clipped.abs().sum(axis=1).replace(0, np.nan)
    over_invested = abs_sum > 1.0
    normaliser = abs_sum.where(over_invested, other=1.0).fillna(1.0)
    return clipped.div(normaliser, axis=0).fillna(0.0)


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio turnover = sum of absolute weight changes."""
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def risk_decomposition(
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling risk contribution per pair (% of total portfolio variance)."""
    pairs = weights.columns
    rc_list = []

    for t in range(window, len(weights)):
        w = weights.iloc[t].values
        r = log_returns.iloc[t - window:t][pairs].values
        cov = np.cov(r.T) * TRADING_DAYS
        port_var = w @ cov @ w
        if port_var < 1e-12:
            rc = np.zeros(len(w))
        else:
            rc = w * (cov @ w) / port_var
        rc_list.append(rc)

    return pd.DataFrame(rc_list, index=weights.index[window:], columns=pairs) * 100


def portfolio_summary(
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    carry_diff: pd.DataFrame,
    cost_per_unit: float = 0.00005,
) -> dict:
    """Compute all portfolio-level statistics."""
    gross    = (weights * log_returns).sum(axis=1)
    carry    = (weights.abs() * carry_diff / TRADING_DAYS).sum(axis=1)
    turnover = compute_turnover(weights)
    costs    = turnover * cost_per_unit
    net      = gross + carry - costs

    return {
        "gross_returns": gross,
        "carry_income":  carry,
        "costs":         costs,
        "net_returns":   net,
        "turnover":      turnover,
        "avg_turnover":  float(turnover.mean() * TRADING_DAYS),
    }
