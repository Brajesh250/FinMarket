"""Tests for validation, metrics, the model registry, training and the signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import DataValidationError, validate_price_frame
from src.evaluation.metrics import baseline_suite, compute_metrics, confusion
from src.evaluation.signal import (
    SignalLabel,
    classify_probability,
    returns_by_signal,
    run_signal_study,
    signal_series,
)
from src.features.builder import build_feature_matrix
from src.models.registry import available_models, build_model, get_model_spec
from src.models.train import run_experiment
from src.utils.stats import annualised_volatility, max_drawdown, summarise_prices


class TestValidation:
    """Structural checks must catch the failure modes that corrupt models."""

    def test_clean_frame_passes(self, synthetic_ohlcv: pd.DataFrame) -> None:
        report = validate_price_frame(synthetic_ohlcv, "TEST")
        assert report.passed
        assert report.rows == len(synthetic_ohlcv)

    def test_empty_frame_fails(self) -> None:
        with pytest.raises(DataValidationError):
            validate_price_frame(pd.DataFrame(), "EMPTY")

    def test_missing_column_fails(self, synthetic_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_price_frame(synthetic_ohlcv.drop(columns=["volume"]), "NOVOL")

    def test_unsorted_index_fails(self, synthetic_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="sorted"):
            validate_price_frame(synthetic_ohlcv.iloc[::-1], "REVERSED")

    def test_duplicate_dates_fail(self, synthetic_ohlcv: pd.DataFrame) -> None:
        duplicated = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[[10]]]).sort_index()
        with pytest.raises(DataValidationError, match="duplicated"):
            validate_price_frame(duplicated, "DUPES")

    def test_negative_price_fails(self, synthetic_ohlcv: pd.DataFrame) -> None:
        broken = synthetic_ohlcv.copy()
        broken.iloc[5, broken.columns.get_loc("low")] = -1.0
        with pytest.raises(DataValidationError, match="non-positive"):
            validate_price_frame(broken, "NEGATIVE")

    def test_high_below_low_fails(self, synthetic_ohlcv: pd.DataFrame) -> None:
        broken = synthetic_ohlcv.copy()
        broken.iloc[7, broken.columns.get_loc("high")] = 0.01
        with pytest.raises(DataValidationError):
            validate_price_frame(broken, "INVERTED")

    def test_too_few_rows_fails(self, synthetic_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="rows"):
            validate_price_frame(synthetic_ohlcv.iloc[:50], "SHORT")

    def test_non_strict_reports_without_raising(self) -> None:
        report = validate_price_frame(pd.DataFrame(), "EMPTY", strict=False)
        assert not report.passed


class TestStats:
    """Descriptive statistics must match hand-checkable values."""

    def test_max_drawdown_of_known_series(self) -> None:
        prices = pd.Series([100.0, 120.0, 60.0, 90.0])
        assert max_drawdown(prices) == pytest.approx(-0.5)

    def test_max_drawdown_of_monotone_rise_is_zero(self) -> None:
        assert max_drawdown(pd.Series([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.0)

    def test_annualised_volatility_scales_by_sqrt_252(self) -> None:
        returns = pd.Series([0.01, -0.01] * 100)
        expected = returns.std(ddof=1) * np.sqrt(252)
        assert annualised_volatility(returns) == pytest.approx(expected)

    def test_summary_fields_are_consistent(self, synthetic_ohlcv: pd.DataFrame) -> None:
        summary = summarise_prices(synthetic_ohlcv)
        assert summary.observations == len(synthetic_ohlcv)
        assert summary.max_drawdown_pct <= 0
        assert summary.largest_daily_gain_pct >= summary.largest_daily_loss_pct
        assert 0 <= summary.up_day_share_pct <= 100

    def test_summary_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            summarise_prices(pd.DataFrame())


class TestMetrics:
    """Metric helpers must behave sensibly at the edges."""

    def test_perfect_prediction(self) -> None:
        y = pd.Series([0, 1, 0, 1, 1, 0])
        metrics = compute_metrics(y, y, y.astype(float))
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)
        assert metrics.roc_auc == pytest.approx(1.0)

    def test_roc_auc_none_for_single_class(self) -> None:
        y = pd.Series([1, 1, 1, 1])
        metrics = compute_metrics(y, y, pd.Series([0.6, 0.7, 0.8, 0.9]))
        assert metrics.roc_auc is None

    def test_confusion_matrix_shape_and_total(self) -> None:
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 0, 1, 1])
        matrix = confusion(y_true, y_pred)
        assert matrix.shape == (2, 2)
        assert matrix.sum() == 4

    def test_baselines_are_computed_on_test_window(self) -> None:
        y_train = pd.Series([1] * 70 + [0] * 30)
        index = pd.bdate_range("2021-01-01", periods=40)
        y_test = pd.Series([1, 0] * 20, index=index)
        daily_return = pd.Series(np.linspace(-0.02, 0.02, 40), index=index)
        baselines = baseline_suite(y_train, y_test, daily_return)
        assert set(baselines) == {"majority_class", "momentum_persistence"}
        # Training majority is class 1, so the constant rule scores the up-day share.
        assert baselines["majority_class"].accuracy == pytest.approx(0.5)


class TestRegistry:
    """The model zoo must be constructible and self-describing."""

    def test_core_models_present(self) -> None:
        specs = available_models()
        for key in ("logistic_regression", "random_forest", "gradient_boosting"):
            assert key in specs

    def test_every_spec_builds(self) -> None:
        for key in available_models():
            pipeline = build_model(key)
            assert hasattr(pipeline, "fit")

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model_spec("not_a_model")


class TestTraining:
    """A full training run must produce coherent, non-degenerate results."""

    @pytest.fixture(scope="class")
    def result(self, synthetic_ohlcv: pd.DataFrame, synthetic_market: pd.Series):
        matrix = build_feature_matrix(synthetic_ohlcv, market_returns=synthetic_market)
        return run_experiment(
            matrix,
            symbol="TEST",
            model_keys=["logistic_regression", "random_forest"],
            run_cv=False,
        )

    def test_all_requested_models_trained(self, result) -> None:
        assert set(result.models) == {"logistic_regression", "random_forest"}

    def test_metrics_are_in_range(self, result) -> None:
        for model in result.models.values():
            assert 0.0 <= model.test_metrics.accuracy <= 1.0
            assert 0.0 <= model.test_metrics.f1 <= 1.0
            if model.test_metrics.roc_auc is not None:
                assert 0.0 <= model.test_metrics.roc_auc <= 1.0

    def test_no_implausibly_perfect_score(self, result) -> None:
        """Near-perfect accuracy on random-walk data would prove leakage."""
        for model in result.models.values():
            assert model.test_metrics.accuracy < 0.75

    def test_probabilities_align_with_test_index(self, result) -> None:
        for model in result.models.values():
            assert model.test_probabilities.index.equals(result.split.X_test.index)
            assert model.test_probabilities.between(0.0, 1.0).all()

    def test_comparison_table_includes_baselines(self, result) -> None:
        table = result.comparison_table()
        assert table["key"].astype(str).str.startswith("baseline_").any()
        assert len(table) == len(result.models) + len(result.baselines)

    def test_best_model_is_a_trained_model(self, result) -> None:
        assert result.best_model.key in result.models


class TestSignal:
    """The research signal must classify and account for costs correctly."""

    def test_thresholds_map_to_labels(self) -> None:
        assert classify_probability(0.80) is SignalLabel.BULLISH
        assert classify_probability(0.50) is SignalLabel.NEUTRAL
        assert classify_probability(0.20) is SignalLabel.BEARISH

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            classify_probability(0.5, bullish_threshold=0.4, bearish_threshold=0.6)

    def test_signal_series_matches_scalar_version(self) -> None:
        probabilities = pd.Series(
            [0.1, 0.5, 0.9], index=pd.bdate_range("2022-01-03", periods=3)
        )
        labels = signal_series(probabilities)
        assert list(labels) == ["Bearish", "Neutral", "Bullish"]

    def test_study_alignment_and_costs(self) -> None:
        index = pd.bdate_range("2022-01-03", periods=60)
        rng = np.random.default_rng(3)
        probabilities = pd.Series(rng.uniform(0.3, 0.7, 60), index=index)
        forward = pd.Series(rng.normal(0.0005, 0.01, 60), index=index)

        study = run_signal_study(probabilities, forward)
        assert study.observations == 60
        assert 0.0 <= study.days_in_market_pct <= 100.0
        assert study.total_cost_pct >= 0.0
        assert study.strategy_max_drawdown_pct <= 0.0

    def test_zero_cost_run_is_not_worse_than_costed_run(self) -> None:
        index = pd.bdate_range("2022-01-03", periods=80)
        rng = np.random.default_rng(11)
        probabilities = pd.Series(rng.uniform(0.4, 0.8, 80), index=index)
        forward = pd.Series(rng.normal(0.001, 0.012, 80), index=index)

        free = run_signal_study(probabilities, forward, transaction_cost_bps=0.0)
        costed = run_signal_study(probabilities, forward, transaction_cost_bps=25.0)
        assert free.strategy_total_return_pct >= costed.strategy_total_return_pct

    def test_non_overlapping_series_raise(self) -> None:
        left = pd.Series([0.6], index=pd.bdate_range("2022-01-03", periods=1))
        right = pd.Series([0.01], index=pd.bdate_range("2023-01-03", periods=1))
        with pytest.raises(ValueError):
            run_signal_study(left, right)

    def test_returns_by_signal_covers_present_buckets(self) -> None:
        index = pd.bdate_range("2022-01-03", periods=50)
        rng = np.random.default_rng(5)
        probabilities = pd.Series(rng.uniform(0.2, 0.8, 50), index=index)
        forward = pd.Series(rng.normal(0, 0.01, 50), index=index)
        table = returns_by_signal(probabilities, forward)
        assert set(table["signal"]) <= {"Bearish", "Neutral", "Bullish"}
        assert table["observations"].sum() == 50


class TestSnapshotData:
    """Checks that run only when the bundled snapshot is present."""

    def test_snapshot_loads_and_validates(self, snapshot_available: bool) -> None:
        if not snapshot_available:
            pytest.skip("bundled snapshot not present in this checkout")
        from src.data.loaders import DataMode, SnapshotProvider, load_prices

        provider = SnapshotProvider()
        symbols = provider.available_symbols()
        assert len(symbols) >= 10

        frame, mode = load_prices(symbols[0], mode=DataMode.SNAPSHOT)
        assert mode is DataMode.SNAPSHOT
        assert frame.index.is_monotonic_increasing
        assert not frame.index.has_duplicates
        assert len(frame) > 1000

    def test_unknown_symbol_raises(self, snapshot_available: bool) -> None:
        if not snapshot_available:
            pytest.skip("bundled snapshot not present in this checkout")
        from src.data.loaders import SnapshotProvider

        with pytest.raises(KeyError):
            SnapshotProvider().fetch("NOT_A_TICKER")
