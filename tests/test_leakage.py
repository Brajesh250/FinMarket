"""Leakage tests.

These are the most important tests in the project. A time-series classifier can
look excellent and be worthless if a single feature quietly contains information
from the future, so each rule is checked directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_feature_matrix
from src.features.technical import (
    average_true_range,
    bollinger_bands,
    macd,
    relative_strength_index,
    rolling_volatility,
    simple_moving_average,
)
from src.models.splits import (
    assert_chronological,
    chronological_split,
    chronological_split_by_date,
)


class TestIndicatorsAreCausal:
    """Truncating the future must not change any past indicator value."""

    @staticmethod
    def _assert_prefix_stable(full: pd.Series, truncated: pd.Series) -> None:
        common = truncated.index
        left = full.loc[common].dropna()
        right = truncated.loc[common].dropna()
        shared = left.index.intersection(right.index)
        assert len(shared) > 50, "not enough overlapping observations to compare"
        np.testing.assert_allclose(
            left.loc[shared].to_numpy(), right.loc[shared].to_numpy(), rtol=1e-10, atol=1e-12
        )

    def test_sma_is_causal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        close = synthetic_ohlcv["close"]
        self._assert_prefix_stable(
            simple_moving_average(close, 20), simple_moving_average(close.iloc[:400], 20)
        )

    def test_rsi_is_causal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        close = synthetic_ohlcv["close"]
        self._assert_prefix_stable(
            relative_strength_index(close), relative_strength_index(close.iloc[:400])
        )

    def test_macd_is_causal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        close = synthetic_ohlcv["close"]
        self._assert_prefix_stable(
            macd(close)["macd_hist"], macd(close.iloc[:400])["macd_hist"]
        )

    def test_bollinger_is_causal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        close = synthetic_ohlcv["close"]
        self._assert_prefix_stable(
            bollinger_bands(close)["bb_position"],
            bollinger_bands(close.iloc[:400])["bb_position"],
        )

    def test_volatility_is_causal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        returns = synthetic_ohlcv["close"].pct_change()
        self._assert_prefix_stable(
            rolling_volatility(returns, 20), rolling_volatility(returns.iloc[:400], 20)
        )

    def test_atr_is_causal(self, synthetic_ohlcv: pd.DataFrame) -> None:
        self._assert_prefix_stable(
            average_true_range(synthetic_ohlcv),
            average_true_range(synthetic_ohlcv.iloc[:400]),
        )


class TestFeatureMatrixIsCausal:
    """The whole feature block must be reproducible from a truncated history."""

    def test_features_unchanged_when_future_removed(
        self, synthetic_ohlcv: pd.DataFrame, synthetic_market: pd.Series
    ) -> None:
        full = build_feature_matrix(synthetic_ohlcv, market_returns=synthetic_market)
        cut = 500
        truncated = build_feature_matrix(
            synthetic_ohlcv.iloc[:cut], market_returns=synthetic_market.iloc[:cut]
        )
        common = full.features.index.intersection(truncated.features.index)
        assert len(common) > 300

        shared_columns = [c for c in full.feature_names if c in truncated.feature_names]
        np.testing.assert_allclose(
            full.features.loc[common, shared_columns].to_numpy(),
            truncated.features.loc[common, shared_columns].to_numpy(),
            rtol=1e-9,
            atol=1e-11,
        )

    def test_no_feature_perfectly_predicts_the_target(
        self, synthetic_ohlcv: pd.DataFrame, synthetic_market: pd.Series
    ) -> None:
        """A near-perfect single-feature correlation is the signature of leakage."""
        matrix = build_feature_matrix(synthetic_ohlcv, market_returns=synthetic_market)
        correlations = matrix.features.corrwith(matrix.target.astype(float)).abs()
        assert correlations.max() < 0.30, (
            f"'{correlations.idxmax()}' correlates {correlations.max():.3f} with the "
            "target, which is far too high for next-day direction and suggests leakage."
        )

    def test_next_day_return_sign_matches_target(
        self, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv)
        assert ((matrix.next_day_return > 0).astype(int) == matrix.target).all()

    def test_features_never_use_same_day_close_of_next_session(
        self, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Perturbing only the final close must not alter any earlier feature row."""
        tampered = synthetic_ohlcv.copy()
        tampered.iloc[-1, tampered.columns.get_loc("close")] *= 1.5

        original = build_feature_matrix(synthetic_ohlcv)
        modified = build_feature_matrix(tampered)
        common = original.features.index.intersection(modified.features.index)
        # The last session's own features may legitimately change; earlier ones must not.
        earlier = common[:-1]
        np.testing.assert_allclose(
            original.features.loc[earlier].to_numpy(),
            modified.features.loc[earlier].to_numpy(),
            rtol=1e-9,
            atol=1e-11,
        )


class TestChronologicalSplitting:
    """Partitions must never overlap in time."""

    def test_blocks_are_strictly_ordered(
        self, synthetic_ohlcv: pd.DataFrame, synthetic_market: pd.Series
    ) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv, market_returns=synthetic_market)
        split = chronological_split(matrix.features, matrix.target)
        assert_chronological(split)
        assert split.X_train.index[-1] < split.X_valid.index[0]
        assert split.X_valid.index[-1] < split.X_test.index[0]

    def test_no_rows_are_lost_or_duplicated(
        self, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv)
        split = chronological_split(matrix.features, matrix.target)
        assert sum(split.sizes.values()) == len(matrix)
        combined = split.X_train.index.union(split.X_valid.index).union(split.X_test.index)
        assert len(combined) == len(matrix)

    def test_shuffled_input_is_rejected(self, synthetic_ohlcv: pd.DataFrame) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv)
        shuffled = matrix.features.sample(frac=1.0, random_state=0)
        with pytest.raises(ValueError, match="sorted by date"):
            chronological_split(shuffled, matrix.target.reindex(shuffled.index))

    def test_misaligned_target_is_rejected(self, synthetic_ohlcv: pd.DataFrame) -> None:
        matrix = build_feature_matrix(synthetic_ohlcv)
        with pytest.raises(ValueError, match="identical index"):
            chronological_split(matrix.features, matrix.target.iloc[:-5])

    def test_panel_split_cuts_on_dates_not_rows(self) -> None:
        """No calendar date may appear in more than one partition."""
        dates = pd.DatetimeIndex(
            np.repeat(pd.bdate_range("2019-01-01", periods=300), 4)
        )
        features = pd.DataFrame(
            {"daily_return": np.linspace(-0.01, 0.01, len(dates)),
             "other": np.arange(len(dates), dtype=float)}
        )
        target = pd.Series((np.arange(len(dates)) % 2).astype(int))

        split, bounds = chronological_split_by_date(features, target, dates)
        assert_chronological(split)

        train_dates = set(split.X_train.index.unique())
        valid_dates = set(split.X_valid.index.unique())
        test_dates = set(split.X_test.index.unique())
        assert not train_dates & valid_dates
        assert not valid_dates & test_dates
        assert not train_dates & test_dates
        assert max(train_dates) < min(valid_dates)
        assert max(valid_dates) < min(test_dates)
        assert bounds["train"][0] <= bounds["train"][1] < bounds["validation"][0]
