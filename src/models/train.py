"""Training and benchmarking orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from config.settings import CV_SPLITS
from src.evaluation.metrics import (
    ClassificationMetrics,
    baseline_suite,
    compute_metrics,
    confusion,
)
from src.features.builder import FeatureMatrix
from src.models.registry import ModelSpec, available_models, build_model
from src.models.splits import (
    ChronologicalSplit,
    assert_chronological,
    chronological_split,
    chronological_split_by_date,
    time_series_cv,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    """A fitted pipeline together with its evaluation results."""

    key: str
    display_name: str
    pipeline: Pipeline
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    cv_roc_auc_mean: float | None
    cv_roc_auc_std: float | None
    test_probabilities: pd.Series
    test_predictions: pd.Series
    confusion_matrix: np.ndarray
    fit_seconds: float

    def as_row(self) -> dict[str, Any]:
        """Flatten the headline numbers into a single record for tables."""
        return {
            "model": self.display_name,
            "key": self.key,
            "val_accuracy": self.validation_metrics.accuracy,
            "val_roc_auc": self.validation_metrics.roc_auc,
            "test_accuracy": self.test_metrics.accuracy,
            "test_precision": self.test_metrics.precision,
            "test_recall": self.test_metrics.recall,
            "test_f1": self.test_metrics.f1,
            "test_roc_auc": self.test_metrics.roc_auc,
            "cv_roc_auc_mean": self.cv_roc_auc_mean,
            "fit_seconds": round(self.fit_seconds, 2),
        }


@dataclass
class ExperimentResult:
    """Everything produced by a single end-to-end training run."""

    symbol: str
    split: ChronologicalSplit
    models: dict[str, TrainedModel]
    baselines: dict[str, ClassificationMetrics]
    feature_names: list[str]
    n_observations: int
    warnings: list[str] = field(default_factory=list)

    @property
    def best_model(self) -> TrainedModel:
        """Model with the highest test ROC-AUC, falling back to accuracy."""
        def score(model: TrainedModel) -> tuple[float, float]:
            return (
                model.test_metrics.roc_auc if model.test_metrics.roc_auc is not None else 0.0,
                model.test_metrics.accuracy,
            )

        return max(self.models.values(), key=score)

    @property
    def best_baseline_accuracy(self) -> float:
        """Accuracy of the strongest naive rule on the same test window."""
        return max(m.accuracy for m in self.baselines.values())

    def comparison_table(self) -> pd.DataFrame:
        """Return a tidy model-comparison table sorted by test ROC-AUC."""
        rows = [model.as_row() for model in self.models.values()]
        for name, metrics in self.baselines.items():
            rows.append(
                {
                    "model": f"Baseline: {name.replace('_', ' ')}",
                    "key": f"baseline_{name}",
                    "val_accuracy": None,
                    "val_roc_auc": None,
                    "test_accuracy": metrics.accuracy,
                    "test_precision": metrics.precision,
                    "test_recall": metrics.recall,
                    "test_f1": metrics.f1,
                    "test_roc_auc": metrics.roc_auc,
                    "cv_roc_auc_mean": None,
                    "fit_seconds": None,
                }
            )
        table = pd.DataFrame(rows)
        return table.sort_values(
            "test_roc_auc", ascending=False, na_position="last"
        ).reset_index(drop=True)


def _predict_proba(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return P(class = 1), tolerating estimators without ``predict_proba``."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    scores = pipeline.decision_function(features)
    return 1.0 / (1.0 + np.exp(-scores))


def train_single_model(
    spec: ModelSpec,
    split: ChronologicalSplit,
    run_cv: bool = True,
    cv_splits: int = CV_SPLITS,
) -> TrainedModel:
    """Fit one model and score it on validation and test windows.

    The model is fitted on the training window, tuned/inspected on validation,
    then refitted on train+validation before the single final test evaluation.
    """
    started = time.perf_counter()
    pipeline = spec.factory()
    pipeline.fit(split.X_train, split.y_train)

    valid_proba = _predict_proba(pipeline, split.X_valid)
    valid_metrics = compute_metrics(
        split.y_valid, (valid_proba >= 0.5).astype(int), valid_proba
    )

    cv_mean: float | None = None
    cv_std: float | None = None
    if run_cv:
        try:
            scores = cross_val_score(
                clone(pipeline),
                split.X_train,
                split.y_train,
                cv=time_series_cv(cv_splits),
                scoring="roc_auc",
                n_jobs=1,
            )
            cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
        except Exception as exc:  # noqa: BLE001 - CV is diagnostic, never fatal
            logger.warning("Cross-validation failed for %s: %s", spec.key, exc)

    # Final refit on all pre-test data, then a single test evaluation.
    X_fit, y_fit = split.train_plus_validation
    final_pipeline = clone(pipeline)
    final_pipeline.fit(X_fit, y_fit)

    test_proba = _predict_proba(final_pipeline, split.X_test)
    test_pred = (test_proba >= 0.5).astype(int)
    test_metrics = compute_metrics(split.y_test, test_pred, test_proba)

    elapsed = time.perf_counter() - started
    logger.info(
        "%-22s val AUC=%s  test AUC=%s  test acc=%.4f  (%.1fs)",
        spec.display_name,
        f"{valid_metrics.roc_auc:.4f}" if valid_metrics.roc_auc else "n/a",
        f"{test_metrics.roc_auc:.4f}" if test_metrics.roc_auc else "n/a",
        test_metrics.accuracy,
        elapsed,
    )

    return TrainedModel(
        key=spec.key,
        display_name=spec.display_name,
        pipeline=final_pipeline,
        validation_metrics=valid_metrics,
        test_metrics=test_metrics,
        cv_roc_auc_mean=cv_mean,
        cv_roc_auc_std=cv_std,
        test_probabilities=pd.Series(test_proba, index=split.X_test.index, name="p_up"),
        test_predictions=pd.Series(test_pred, index=split.X_test.index, name="prediction"),
        confusion_matrix=confusion(split.y_test, test_pred),
        fit_seconds=elapsed,
    )


def run_experiment(
    matrix: FeatureMatrix,
    symbol: str,
    model_keys: list[str] | None = None,
    run_cv: bool = True,
) -> ExperimentResult:
    """Train every requested model on one ticker and benchmark against baselines.

    Args:
        matrix: Model-ready dataset from :func:`build_feature_matrix`.
        symbol: Ticker being modelled, for labelling only.
        model_keys: Subset of registry keys; defaults to every available model.
        run_cv: Whether to run expanding-window cross-validation.

    Returns:
        A populated :class:`ExperimentResult`.
    """
    split = chronological_split(matrix.features, matrix.target)
    assert_chronological(split)
    logger.info("Split for %s: %s | periods %s", symbol, split.sizes, split.periods)

    specs = available_models()
    keys = model_keys or list(specs)
    warnings: list[str] = []

    trained: dict[str, TrainedModel] = {}
    for key in keys:
        if key not in specs:
            warnings.append(f"model '{key}' unavailable in this environment")
            continue
        try:
            trained[key] = train_single_model(specs[key], split, run_cv=run_cv)
        except Exception as exc:  # noqa: BLE001 - one bad model must not stop the run
            logger.exception("Training failed for %s", key)
            warnings.append(f"{key} failed to train: {exc}")

    if not trained:
        raise RuntimeError("No model trained successfully.")

    daily_return_test = matrix.features.loc[split.X_test.index, "daily_return"]
    baselines = baseline_suite(split.y_train, split.y_test, daily_return_test)

    return ExperimentResult(
        symbol=symbol,
        split=split,
        models=trained,
        baselines=baselines,
        feature_names=matrix.feature_names,
        n_observations=len(matrix),
        warnings=warnings,
    )


def run_panel_experiment(
    panel,
    model_keys: list[str] | None = None,
    run_cv: bool = False,
    label: str = "PANEL",
) -> tuple[ExperimentResult, dict[str, tuple[str, str]]]:
    """Train and benchmark every model on a stacked multi-ticker panel.

    Args:
        panel: A :class:`src.features.panel.PanelDataset`.
        model_keys: Subset of registry keys; defaults to every available model.
        run_cv: Whether to run expanding-window cross-validation. Off by
            default because the panel is large and CV is expensive.
        label: Name used in reporting.

    Returns:
        The experiment result and the calendar bounds of each partition.
    """
    split, bounds = chronological_split_by_date(panel.features, panel.target, panel.dates)
    assert_chronological(split)
    logger.info("Panel split %s | periods %s", split.sizes, bounds)

    specs = available_models()
    keys = model_keys or list(specs)
    warnings: list[str] = []
    trained: dict[str, TrainedModel] = {}

    for key in keys:
        if key not in specs:
            warnings.append(f"model '{key}' unavailable in this environment")
            continue
        try:
            trained[key] = train_single_model(specs[key], split, run_cv=run_cv)
        except Exception as exc:  # noqa: BLE001 - keep going if one model fails
            logger.exception("Panel training failed for %s", key)
            warnings.append(f"{key} failed to train: {exc}")

    if not trained:
        raise RuntimeError("No model trained successfully on the panel.")

    baselines = baseline_suite(
        split.y_train, split.y_test, split.X_test["daily_return"]
    )
    result = ExperimentResult(
        symbol=label,
        split=split,
        models=trained,
        baselines=baselines,
        feature_names=panel.feature_names,
        n_observations=len(panel),
        warnings=warnings,
    )
    return result, bounds
