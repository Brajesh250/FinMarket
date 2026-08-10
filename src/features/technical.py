"""Technical indicators implemented directly on pandas.

Every function here is *causal*: the value at row ``t`` depends only on rows
``<= t``. That property is what makes the downstream supervised problem
leakage-free, and it is asserted in ``tests/test_leakage.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_WINDOW,
)


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Rolling arithmetic mean over ``window`` observations."""
    return series.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Exponentially weighted mean with span ``window``."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def relative_strength_index(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Wilder's Relative Strength Index.

    RSI compares the size of recent gains to recent losses on a 0-100 scale.
    Readings above 70 are conventionally described as "overbought" and below 30
    as "oversold"; here it is simply used as a momentum feature.

    Args:
        close: Close price series.
        window: Look-back length, 14 by default.

    Returns:
        Series of RSI values in ``[0, 100]``.
    """
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing is an EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # A window with zero losses is maximally strong; zero gains is minimally so.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns a frame with three columns: ``macd`` (fast EMA minus slow EMA),
    ``macd_signal`` (EMA of the MACD line) and ``macd_hist`` (their difference).
    A positive histogram means short-term momentum is running ahead of the
    longer-term trend.
    """
    price = close.astype(float)
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}
    )


def bollinger_bands(
    close: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_STD,
) -> pd.DataFrame:
    """Bollinger Bands and the position of price within them.

    ``bb_position`` is 0 at the lower band and 1 at the upper band, so it gives
    a scale-free measure of how stretched the price is relative to its own
    recent range.
    """
    price = close.astype(float)
    middle = price.rolling(window=window, min_periods=window).mean()
    spread = price.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * spread
    lower = middle - num_std * spread
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": (upper - lower) / middle,
            "bb_position": (price - lower) / width,
        }
    )


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of returns (not annualised)."""
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def average_true_range(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range, a range-based volatility measure."""
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    prev_close = frame["close"].astype(float).shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
