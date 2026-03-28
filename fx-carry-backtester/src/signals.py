"""
signals.py — Signal generation for FX carry + momentum strategy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def carry_signal(carry_diff: pd.DataFrame) -> pd.DataFrame:
    """+1 if carry positive, -1 if negative, 0 if below 25bps threshold."""
    threshold = 0.0025
    sig = pd.DataFrame(0.0, index=carry_diff.index, columns=carry_diff.columns)
    sig[carry_diff >  threshold] =  1.0
    sig[carry_diff < -threshold] = -1.0
    return sig


def momentum_signal(log_returns: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """+1 if trailing return over lookback days is positive, -1 otherwise."""
    cum_return = log_returns.rolling(lookback).sum()
    return np.sign(cum_return).fillna(0.0)


def combined_signal(
    carry_diff: pd.DataFrame,
    log_returns: pd.DataFrame,
    momentum_lookback: int = 63,
    carry_weight: float = 0.5,
    momentum_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Combined carry + momentum signal.
    Filter mode (default): carry signal when both agree, flat when they disagree.
    Shifted 1 day to avoid look-ahead bias.
    """
    c_sig = carry_signal(carry_diff)
    m_sig = momentum_signal(log_returns, momentum_lookback)

    if abs(carry_weight - 0.5) < 1e-9 and abs(momentum_weight - 0.5) < 1e-9:
        signal = c_sig.copy()
        signal[c_sig * m_sig < 0] = 0.0
        signal[m_sig == 0.0] = 0.0
    else:
        signal = (carry_weight * c_sig + momentum_weight * m_sig).clip(-1.0, 1.0)

    return signal.shift(1).fillna(0.0)


def rank_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Normalise signals so absolute weights sum to 1."""
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    return signal.div(abs_sum, axis=0).fillna(0.0)
