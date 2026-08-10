"""Descriptive statistics used across the analytics and reporting layers.

These are deliberately simple, textbook definitions that a reader with a basic
finance background can verify by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class PriceSummary:
    """Headline statistics for a single price series."""

    observations: int
    start_date: str
    end_date: str
    total_return_pct: float
    annualised_return_pct: float
    annualised_volatility_pct: float
    largest_daily_gain_pct: float
    largest_daily_loss_pct: float
    max_drawdown_pct: float
    up_day_share_pct: float

    def as_dict(self) -> dict[str, Any]:
        """Return the summary as a plain dictionary."""
        return asdict(self)


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple period-over-period percentage change of a close series."""
    return close.astype(float).pct_change()


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline of a price series, as a fraction.

    Args:
        close: Price series ordered chronologically.

    Returns:
        A non-positive float, e.g. ``-0.32`` for a 32% drawdown. Returns 0.0
        for an empty or single-observation series.
    """
    prices = close.astype(float).dropna()
    if len(prices) < 2:
        return 0.0
    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1.0
    return float(drawdown.min())


def annualised_return(returns: pd.Series) -> float:
    """Geometric mean daily return scaled to a year."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    growth = float((1.0 + clean).prod())
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS_PER_YEAR / len(clean)) - 1.0


def annualised_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns scaled by sqrt(252)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def summarise_prices(frame: pd.DataFrame) -> PriceSummary:
    """Compute headline statistics for an OHLCV frame.

    Args:
        frame: Frame indexed by date containing at least a ``close`` column.

    Returns:
        A populated :class:`PriceSummary`.

    Raises:
        ValueError: If ``frame`` is empty or lacks a ``close`` column.
    """
    if frame.empty:
        raise ValueError("Cannot summarise an empty price frame.")
    if "close" not in frame.columns:
        raise ValueError("Price frame must contain a 'close' column.")

    close = frame["close"].astype(float)
    returns = daily_returns(close).dropna()
    total_return = float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close) > 1 else 0.0
    up_share = float((returns > 0).mean()) if not returns.empty else 0.0

    return PriceSummary(
        observations=int(len(frame)),
        start_date=str(pd.Timestamp(frame.index[0]).date()),
        end_date=str(pd.Timestamp(frame.index[-1]).date()),
        total_return_pct=round(total_return * 100, 2),
        annualised_return_pct=round(annualised_return(returns) * 100, 2),
        annualised_volatility_pct=round(annualised_volatility(returns) * 100, 2),
        largest_daily_gain_pct=round(float(returns.max()) * 100, 2) if not returns.empty else 0.0,
        largest_daily_loss_pct=round(float(returns.min()) * 100, 2) if not returns.empty else 0.0,
        max_drawdown_pct=round(max_drawdown(close) * 100, 2),
        up_day_share_pct=round(up_share * 100, 2),
    )


def monthly_return_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot daily closes into a year x month percentage-return table."""
    close = frame["close"].astype(float)
    monthly = close.resample("ME").last().pct_change().dropna() * 100
    table = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.strftime("%b"),
            "return_pct": monthly.to_numpy(),
        }
    )
    pivot = table.pivot_table(index="year", columns="month", values="return_pct")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot.reindex(columns=[m for m in month_order if m in pivot.columns])


def returns_by_weekday(frame: pd.DataFrame) -> pd.DataFrame:
    """Average daily return grouped by day of week."""
    returns = daily_returns(frame["close"]).dropna() * 100
    grouped = returns.groupby(returns.index.day_name()).agg(["mean", "count"])
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    grouped = grouped.reindex([d for d in order if d in grouped.index])
    grouped.columns = ["mean_return_pct", "observations"]
    return grouped.reset_index(names="weekday")
