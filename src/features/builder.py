"""Assembly of the supervised learning matrix.

The contract enforced throughout this module:

* every column in ``X`` for session ``t`` is computable from data published on
  or before the close of session ``t``;
* the label ``y`` for session ``t`` describes what happens on session ``t+1``.

That single shift is the only place where future information is touched, and it
is applied to the target alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import (
    EMA_WINDOWS,
    MOMENTUM_WINDOWS,
    SMA_WINDOWS,
    VOLATILITY_WINDOWS,
    VOLUME_SMA_WINDOW,
    WARMUP_ROWS,
)
from src.features.technical import (
    average_true_range,
    bollinger_bands,
    exponential_moving_average,
    macd,
    relative_strength_index,
    rolling_volatility,
    simple_moving_average,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "target_up_next_day"
# Columns kept for analysis/plotting but never fed to a model.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "adj_close", "volume",
    TARGET_COLUMN, "next_day_return",
)


@dataclass(frozen=True)
class FeatureMatrix:
    """A model-ready dataset plus the context needed to interpret it."""

    features: pd.DataFrame
    target: pd.Series
    next_day_return: pd.Series
    prices: pd.DataFrame
    feature_names: list[str]

    def __len__(self) -> int:
        """Number of usable observations."""
        return len(self.features)

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last date present in the matrix."""
        return self.features.index[0], self.features.index[-1]


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return raw-price-derived features (returns, ranges, gaps)."""
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)

    out = pd.DataFrame(index=frame.index)
    out["daily_return"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))
    # Overnight: previous close to today's open. Intraday: open to close.
    out["overnight_return"] = open_ / close.shift(1) - 1.0
    out["intraday_return"] = close / open_ - 1.0
    out["high_low_range"] = (high - low) / close.replace(0.0, np.nan)
    out["close_to_high"] = (high - close) / close.replace(0.0, np.nan)
    return out


def build_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return multi-horizon trailing returns."""
    close = frame["close"].astype(float)
    out = pd.DataFrame(index=frame.index)
    for window in MOMENTUM_WINDOWS:
        out[f"return_{window}d"] = close.pct_change(window)
    return out


def build_moving_average_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return moving-average *ratios* rather than raw levels.

    Feeding raw prices to a classifier makes the model non-stationary and
    ticker-specific. Ratios such as ``close / SMA20`` are scale-free and
    comparable across symbols and eras.
    """
    close = frame["close"].astype(float)
    out = pd.DataFrame(index=frame.index)

    smas = {w: simple_moving_average(close, w) for w in SMA_WINDOWS}
    emas = {w: exponential_moving_average(close, w) for w in EMA_WINDOWS}

    for window, series in smas.items():
        out[f"close_to_sma{window}"] = close / series - 1.0
    for window, series in emas.items():
        out[f"close_to_ema{window}"] = close / series - 1.0

    if 5 in smas and 20 in smas:
        out["sma5_to_sma20"] = smas[5] / smas[20] - 1.0
    if 20 in smas and 50 in smas:
        out["sma20_to_sma50"] = smas[20] / smas[50] - 1.0
    return out


def build_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return RSI, MACD, Bollinger and ATR features."""
    close = frame["close"].astype(float)
    out = pd.DataFrame(index=frame.index)
    out["rsi_14"] = relative_strength_index(close)

    macd_frame = macd(close)
    # Normalise MACD by price so the scale is comparable across tickers.
    out["macd"] = macd_frame["macd"] / close
    out["macd_signal"] = macd_frame["macd_signal"] / close
    out["macd_hist"] = macd_frame["macd_hist"] / close

    bands = bollinger_bands(close)
    out["bb_position"] = bands["bb_position"]
    out["bb_width"] = bands["bb_width"]

    out["atr_14_norm"] = average_true_range(frame) / close
    return out


def build_volatility_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rolling realised-volatility features."""
    returns = frame["close"].astype(float).pct_change()
    out = pd.DataFrame(index=frame.index)
    for window in VOLATILITY_WINDOWS:
        out[f"volatility_{window}d"] = rolling_volatility(returns, window)
    if 5 in VOLATILITY_WINDOWS and 20 in VOLATILITY_WINDOWS:
        out["vol_ratio_5_20"] = (
            out["volatility_5d"] / out["volatility_20d"].replace(0.0, np.nan)
        )
    return out


def build_volume_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return volume dynamics relative to the ticker's own recent norm."""
    volume = frame["volume"].astype(float)
    out = pd.DataFrame(index=frame.index)
    out["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    volume_sma = volume.rolling(VOLUME_SMA_WINDOW, min_periods=VOLUME_SMA_WINDOW).mean()
    out["relative_volume"] = volume / volume_sma.replace(0.0, np.nan)
    out["volume_trend_5d"] = (
        volume.rolling(5, min_periods=5).mean() / volume_sma.replace(0.0, np.nan)
    )
    return out


def build_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return simple seasonality flags known in advance."""
    index = frame.index
    out = pd.DataFrame(index=index)
    out["day_of_week"] = index.dayofweek.astype(float)
    out["month"] = index.month.astype(float)
    return out


def build_market_features(
    frame: pd.DataFrame, market_returns: pd.Series | None
) -> pd.DataFrame:
    """Return broad-market context aligned to the ticker's sessions."""
    out = pd.DataFrame(index=frame.index)
    if market_returns is None or market_returns.dropna().empty:
        return out

    aligned = market_returns.reindex(frame.index)
    coverage = float(aligned.notna().mean())
    if coverage < 0.5:
        logger.warning("Market series covers only %.0f%% of sessions; skipping.", coverage * 100)
        return out

    out["market_return"] = aligned
    out["market_return_5d"] = aligned.rolling(5, min_periods=5).sum()
    out["market_volatility_20d"] = aligned.rolling(20, min_periods=20).std(ddof=1)
    # Excess return: how the stock did relative to the market on the same day.
    out["excess_return"] = frame["close"].astype(float).pct_change() - aligned
    return out


def build_target(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the binary next-day direction label and the raw next-day return.

    ``target = 1`` when ``close(t+1) > close(t)``, else 0. The final row of any
    frame has no ``t+1`` and is therefore dropped downstream.
    """
    close = frame["close"].astype(float)
    next_day_return = close.shift(-1) / close - 1.0
    target = (next_day_return > 0).astype("Int64")
    target = target.where(next_day_return.notna())
    return target.rename(TARGET_COLUMN), next_day_return.rename("next_day_return")


def build_feature_matrix(
    frame: pd.DataFrame,
    market_returns: pd.Series | None = None,
    warmup: int = WARMUP_ROWS,
) -> FeatureMatrix:
    """Build the full model-ready dataset from an OHLCV frame.

    Args:
        frame: Validated OHLCV frame indexed by date.
        market_returns: Optional broad-market daily returns for context features.
        warmup: Leading rows discarded so every indicator is fully formed.

    Returns:
        A :class:`FeatureMatrix`.

    Raises:
        ValueError: If nothing survives cleaning.
    """
    blocks = [
        build_price_features(frame),
        build_momentum_features(frame),
        build_moving_average_features(frame),
        build_technical_features(frame),
        build_volatility_features(frame),
        build_volume_features(frame),
        build_calendar_features(frame),
        build_market_features(frame, market_returns),
    ]
    features = pd.concat(blocks, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)

    target, next_day_return = build_target(frame)

    combined = features.copy()
    combined[TARGET_COLUMN] = target
    combined["next_day_return"] = next_day_return

    if warmup > 0:
        combined = combined.iloc[warmup:]
    combined = combined.dropna()

    if combined.empty:
        raise ValueError(
            "No usable rows after feature construction; the input series is too short."
        )

    feature_names = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
    matrix = FeatureMatrix(
        features=combined[feature_names].astype(float),
        target=combined[TARGET_COLUMN].astype(int),
        next_day_return=combined["next_day_return"].astype(float),
        prices=frame.loc[combined.index],
        feature_names=feature_names,
    )
    logger.info(
        "Built feature matrix: %d rows x %d features (%s to %s)",
        len(matrix),
        len(feature_names),
        matrix.date_range[0].date(),
        matrix.date_range[1].date(),
    )
    return matrix
