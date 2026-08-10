"""Tests for technical indicators and feature assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import (
    TARGET_COLUMN,
    build_feature_matrix,
    build_moving_average_features,
    build_target,
)
from src.features.technical import (
    bollinger_bands,
    exponential_moving_average,
    macd,
    relative_strength_index,
    simple_moving_average,
)


class TestMovingAverages:
    """Moving averages must match their textbook definitions."""

    def test_sma_matches_manual_mean(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = simple_moving_average(series, 3)
        assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[5] == pytest.approx(5.0)

    def test_sma_warmup_is_nan(self) -> None:
        series = pd.Series(np.arange(20, dtype=float))
        assert simple_moving_average(series, 10).iloc[:9].isna().all()

    def test_ema_reacts_faster_than_sma(self, synthetic_ohlcv: pd.DataFrame) -> None:
        close = synthetic_ohlcv["close"]
        ema = exponential_moving_average(close, 20).dropna()
        sma = simple_moving_average(close, 20).dropna()
        common = ema.index.intersection(sma.index)
        # A faster-reacting average tracks price more closely on average.
        ema_error = (close.loc[common] - ema.loc[common]).abs().mean()
        sma_error = (close.loc[common] - sma.loc[common]).abs().mean()
        assert ema_error < sma_error


class TestRSI:
    """RSI must stay bounded and respond correctly to monotone inputs."""

    def test_bounded_between_zero_and_hundred(self, synthetic_ohlcv: pd.DataFrame) -> None:
        rsi = relative_strength_index(synthetic_ohlcv["close"]).dropna()
        assert not rsi.empty
        assert rsi.between(0.0, 100.0).all()

    def test_monotonic_rise_gives_maximum(self) -> None:
        rising = pd.Series(np.linspace(10, 60, 80))
        assert relative_strength_index(rising).dropna().iloc[-1] == pytest.approx(100.0)

    def test_monotonic_fall_gives_minimum(self) -> None:
        falling = pd.Series(np.linspace(60, 10, 80))
        assert relative_strength_index(falling).dropna().iloc[-1] == pytest.approx(0.0)


class TestMACD:
    """MACD components must be internally consistent."""

    def test_histogram_is_line_minus_signal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        frame = macd(synthetic_ohlcv["close"]).dropna()
        expected = frame["macd"] - frame["macd_signal"]
        pd.testing.assert_series_equal(
            frame["macd_hist"], expected, check_names=False, rtol=1e-9
        )

    def test_columns_present(self, synthetic_ohlcv: pd.DataFrame) -> None:
        assert list(macd(synthetic_ohlcv["close"]).columns) == [
            "macd", "macd_signal", "macd_hist"
        ]


class TestBollinger:
    """Bands must bracket the middle line and position must be well scaled."""

    def test_upper_above_middle_above_lower(self, synthetic_ohlcv: pd.DataFrame) -> None:
        bands = bollinger_bands(synthetic_ohlcv["close"]).dropna()
        assert (bands["bb_upper"] >= bands["bb_middle"]).all()
        assert (bands["bb_middle"] >= bands["bb_lower"]).all()

    def test_position_mostly_within_unit_interval(self, synthetic_ohlcv: pd.DataFrame) -> None:
        position = bollinger_bands(synthetic_ohlcv["close"])["bb_position"].dropna()
        # A two-standard-deviation envelope estimated on a rolling 20-session window
        # contains most but not all observations; the theoretical 95% is not reached
        # because the standard deviation itself is estimated from only 20 points.
        assert 0.80 < position.between(0.0, 1.0).mean() < 0.99


class TestTarget:
    """The label must describe the next session and nothing else."""

    def test_target_is_next_day_direction(self) -> None:
        frame = pd.DataFrame(
            {"close": [10.0, 11.0, 10.5, 12.0]},
            index=pd.bdate_range("2020-01-01", periods=4),
        )
        target, forward = build_target(frame)
        assert target.iloc[0] == 1   # 10 -> 11 is up
        assert target.iloc[1] == 0   # 11 -> 10.5 is down
        assert target.iloc[2] == 1   # 10.5 -> 12 is up
        assert pd.isna(target.iloc[3])  # no t+1 exists
        assert forward.iloc[0] == pytest.approx(0.1)

    def test_flat_close_counts_as_down(self) -> None:
        frame = pd.DataFrame(
            {"close": [10.0, 10.0, 10.0]}, index=pd.bdate_range("2020-01-01", periods=3)
        )
        target, _ = build_target(frame)
        assert target.iloc[0] == 0


class TestFeatureMatrix:
    """The assembled matrix must be clean, aligned and finite."""

    def test_matrix_is_finite_and_aligned(self, synthetic_ohlcv, synthetic_market) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv, market_returns=synthetic_market)
        assert len(matrix) > 500
        assert np.isfinite(matrix.features.to_numpy()).all()
        assert matrix.features.index.equals(matrix.target.index)
        assert matrix.features.index.equals(matrix.next_day_return.index)
        assert matrix.features.index.is_monotonic_increasing

    def test_target_not_present_among_features(self, synthetic_ohlcv) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv)
        assert TARGET_COLUMN not in matrix.feature_names
        assert "next_day_return" not in matrix.feature_names

    def test_raw_prices_excluded_from_features(self, synthetic_ohlcv) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv)
        for column in ("open", "high", "low", "close", "adj_close", "volume"):
            assert column not in matrix.feature_names

    def test_market_features_present_when_supplied(self, synthetic_ohlcv, synthetic_market) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv, market_returns=synthetic_market)
        assert "market_return" in matrix.feature_names

    def test_market_features_absent_when_missing(self, synthetic_ohlcv) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv, market_returns=None)
        assert "market_return" not in matrix.feature_names

    def test_moving_average_features_are_ratios(self, synthetic_ohlcv) -> None:
        frame = build_moving_average_features(synthetic_ohlcv).dropna()
        # Ratio features are centred near zero, unlike raw price levels.
        assert frame["close_to_sma20"].abs().max() < 1.0

    def test_short_series_raises(self) -> None:
        short = pd.DataFrame(
            {
                "open": [1.0] * 20, "high": [1.1] * 20, "low": [0.9] * 20,
                "close": [1.0] * 20, "volume": [100.0] * 20,
            },
            index=pd.bdate_range("2020-01-01", periods=20),
        )
        with pytest.raises(ValueError):
            build_feature_matrix(short)
