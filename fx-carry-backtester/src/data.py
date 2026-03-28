"""
data.py — Download FX spot rates and interest rate differentials.
"""
from __future__ import annotations
import yfinance as yf
import pandas as pd
import numpy as np

FX_PAIRS: dict[str, tuple[str, str]] = {
    "AUDJPY=X": ("AUD", "JPY"),
    "NZDJPY=X": ("NZD", "JPY"),
    "GBPJPY=X": ("GBP", "JPY"),
    "AUDUSD=X": ("AUD", "USD"),
    "NZDUSD=X": ("NZD", "USD"),
    "USDCHF=X": ("USD", "CHF"),
    "USDJPY=X": ("USD", "JPY"),
    "EURUSD=X": ("EUR", "USD"),
}

RATE_SCHEDULE: dict[str, list[tuple[str, float]]] = {
    "AUD": [("2015-01-01", 2.50), ("2016-05-04", 1.75), ("2019-06-05", 1.25),
            ("2020-03-20", 0.25), ("2022-05-04", 0.85), ("2022-11-02", 2.85),
            ("2023-05-03", 3.85), ("2024-01-01", 4.35)],
    "NZD": [("2015-01-01", 3.50), ("2016-03-10", 2.25), ("2019-08-08", 1.00),
            ("2020-03-16", 0.25), ("2022-02-23", 1.00), ("2022-10-05", 3.50),
            ("2023-05-24", 5.50), ("2024-01-01", 5.50)],
    "GBP": [("2015-01-01", 0.50), ("2016-08-04", 0.25), ("2018-08-02", 0.75),
            ("2020-03-19", 0.10), ("2022-05-05", 1.00), ("2022-11-03", 3.00),
            ("2023-08-03", 5.25), ("2024-08-01", 5.00)],
    "EUR": [("2015-01-01", 0.05), ("2016-03-16", 0.00), ("2022-07-27", 0.50),
            ("2022-11-02", 2.00), ("2023-06-21", 4.00), ("2024-06-12", 3.75)],
    "USD": [("2015-01-01", 0.25), ("2015-12-17", 0.50), ("2018-09-27", 2.25),
            ("2019-10-31", 1.50), ("2020-03-16", 0.25), ("2022-03-17", 0.50),
            ("2022-09-22", 3.25), ("2023-05-04", 5.25), ("2024-09-19", 5.00)],
    "JPY": [("2015-01-01", 0.10), ("2016-01-29", -0.10), ("2024-03-19", 0.10),
            ("2024-07-31", 0.25)],
    "CHF": [("2015-01-01", -0.75), ("2022-06-16", -0.25), ("2022-09-22", 0.50),
            ("2023-06-22", 1.75), ("2024-03-21", 1.50)],
}


def _build_rate_series(schedule: list[tuple[str, float]], index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.to_datetime([d for d, _ in schedule])
    rates = [r for _, r in schedule]
    s = pd.Series(rates, index=dates)
    return s.reindex(index, method="ffill").ffill().fillna(0.0) / 100.0


def download_fx_prices(start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download daily closing prices for all pairs in FX_PAIRS."""
    tickers = list(FX_PAIRS.keys())
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"].copy()
    close.columns = [t.replace("=X", "") for t in close.columns]
    return close.ffill().dropna(how="all")


def build_carry_differential(index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each pair, compute annualised carry = rate(base) - rate(quote)."""
    carry = {}
    for ticker, (base, quote) in FX_PAIRS.items():
        name = ticker.replace("=X", "")
        base_rate  = _build_rate_series(RATE_SCHEDULE.get(base,  [("2015-01-01", 0.0)]), index)
        quote_rate = _build_rate_series(RATE_SCHEDULE.get(quote, [("2015-01-01", 0.0)]), index)
        carry[name] = base_rate - quote_rate
    return pd.DataFrame(carry, index=index)
