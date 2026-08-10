"""Model explainability: which inputs actually drive the predictions.

Three complementary views are supported, in order of preference:

1. native tree feature importances,
2. logistic-regression coefficients (signed, so direction is visible),
3. permutation importance, which works for any estimator and is measured
   against a held-out window rather than the training data.

SHAP is used opportunistically when the package is installed and the estimator
is tree-based; it is never a hard requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config.settings import RANDOM_STATE
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _final_estimator(pipeline: Pipeline):
    """Return the estimator at the end of a pipeline."""
    return pipeline.steps[-1][1]


def native_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    """Return native importances or coefficients, if the estimator exposes them.

    Returns:
        A frame with ``feature``, ``importance`` and ``signed_value`` columns,
        sorted by absolute importance, or None when unsupported.
    """
    estimator = _final_estimator(pipeline)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        return (
            pd.DataFrame(
                {"feature": feature_names, "importance": values, "signed_value": values}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float).ravel()
        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": np.abs(coefficients),
                    "signed_value": coefficients,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return None


def permutation_feature_importance(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Measure importance by shuffling each column on a held-out window.

    This is the most honest of the three views: it reports how much predictive
    performance is genuinely lost when a feature's information is destroyed.
    """
    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(features.columns),
                "importance": result.importances_mean,
                "std": result.importances_std,
                "signed_value": result.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def top_features(
    pipeline: Pipeline,
    feature_names: list[str],
    features: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, str]:
    """Return the ``top_n`` most influential features and the method used.

    Falls back from native importance to permutation importance automatically.
    """
    frame = native_importance(pipeline, feature_names)
    method = "native"

    if frame is None:
        if features is None or target is None:
            raise ValueError(
                "This estimator exposes no native importances; supply features "
                "and target so permutation importance can be computed."
            )
        frame = permutation_feature_importance(pipeline, features, target)
        method = "permutation"

    return frame.head(top_n).copy(), method


def shap_summary(
    pipeline: Pipeline, features: pd.DataFrame, max_samples: int = 300
) -> pd.DataFrame | None:
    """Return mean absolute SHAP values per feature, when SHAP is usable.

    Returns None (and logs) if ``shap`` is missing or the estimator is not
    supported by ``TreeExplainer`` — this keeps SHAP strictly optional.
    """
    try:
        import shap
    except ImportError:
        logger.info("shap not installed; skipping SHAP summary.")
        return None

    estimator = _final_estimator(pipeline)
    if not hasattr(estimator, "feature_importances_"):
        logger.info("SHAP TreeExplainer needs a tree model; skipping.")
        return None

    sample = features.iloc[-max_samples:] if len(features) > max_samples else features
    transformed = sample
    if len(pipeline.steps) > 1:
        transformed = pd.DataFrame(
            pipeline[:-1].transform(sample), columns=sample.columns, index=sample.index
        )

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # (n_samples, n_features, n_classes)
            values = values[:, :, -1]
        magnitude = np.abs(values).mean(axis=0)
    except Exception as exc:  # noqa: BLE001 - SHAP is best-effort only
        logger.warning("SHAP computation failed: %s", exc)
        return None

    return (
        pd.DataFrame({"feature": list(sample.columns), "mean_abs_shap": magnitude})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def feature_correlation(features: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the correlation matrix of the ``top_n`` highest-variance features."""
    if features.shape[1] <= top_n:
        subset = features
    else:
        subset = features[features.std().sort_values(ascending=False).head(top_n).index]
    return subset.corr()
