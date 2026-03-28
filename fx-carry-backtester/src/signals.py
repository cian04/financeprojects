"""
signals.py — Signal generation for FX carry + momentum strategy.

Improvements v2:
- Composite momentum signal (1M + 3M + 12M weighted average)
- Minimum carry threshold raised to 1% (filter weak carry environments)
- Volatility regime filter (flat when portfolio vol > 15%)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def carry_signal(carry_diff: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    +1 if carry > threshold, -1 if carry < -threshold, 0 otherwise.
    Threshold raised to 1% to filter weak carry environments (2015-2019 ZIRP).
    """
    sig = pd.DataFrame(0.0, index=carry_diff.index, columns=carry_diff.columns)
    sig[carry_diff >  threshold] =  1.0
    sig[carry_diff < -threshold] = -1.0
    return sig


def momentum_signal(log_returns: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Simple momentum: sign of trailing return over lookback days."""
    return np.sign(log_returns.rolling(lookback).sum()).fillna(0.0)


def composite_momentum_signal(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Composite momentum: weighted average of 1M (21d), 3M (63d), 12M (252d).
    Weights: 0.25 / 0.50 / 0.25 — 3M dominates, 12M adds trend filter.
    More robust than single-lookback momentum across different market regimes.
    """
    m1  = np.sign(log_returns.rolling(21).sum()).fillna(0.0)
    m3  = np.sign(log_returns.rolling(63).sum()).fillna(0.0)
    m12 = np.sign(log_returns.rolling(252).sum()).fillna(0.0)
    composite = 0.25 * m1 + 0.50 * m3 + 0.25 * m12
    return composite


def vol_regime_filter(
    log_returns: pd.DataFrame,
    weights: pd.DataFrame,
    vol_cap: float = 0.15,
    window: int = 21,
) -> pd.Series:
    """
    Returns 1.0 (normal) or 0.0 (flat) based on rolling portfolio vol.
    If realised vol > vol_cap, set all signals to zero.
    Protects against carry unwinds in high-vol regimes (COVID 2020, Aug 2024 JPY).
    Shifted 1 day to avoid look-ahead bias.
    """
    port_ret = (weights * log_returns).sum(axis=1)
    rolling_vol = port_ret.rolling(window).std() * np.sqrt(252)
    active = (rolling_vol <= vol_cap).astype(float)
    return active.shift(1).fillna(1.0)


def combined_signal(
    carry_diff: pd.DataFrame,
    log_returns: pd.DataFrame,
    carry_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Combined carry + composite momentum signal.
    - Flat when signals disagree
    - Flat when carry differential below threshold
    - Shifted 1 day to avoid look-ahead bias
    """
    c_sig = carry_signal(carry_diff, carry_threshold)
    m_sig = composite_momentum_signal(log_returns)

    signal = c_sig.copy()
    # Flat when momentum contradicts carry
    signal[c_sig * m_sig < 0] = 0.0
    # Flat when no momentum data yet
    signal[m_sig == 0.0] = 0.0

    return signal.shift(1).fillna(0.0)


def rank_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Normalise signals so absolute weights sum to 1."""
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    return signal.div(abs_sum, axis=0).fillna(0.0)
